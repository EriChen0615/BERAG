"""
Compatibility layer for transformers KV cache APIs.

- Legacy API (e.g. older / vendored transformers): DynamicCache has
  .key_cache and .value_cache (lists of tensors). Index and append directly.
- New API (e.g. transformers >= 4.46): Cache is a container of layers;
  each layer has .keys and .values. No .key_cache / .value_cache on the cache.
"""

import torch


def _is_legacy_cache(cache) -> bool:
    """True if cache uses key_cache / value_cache (legacy DynamicCache)."""
    return hasattr(cache, "key_cache")


def _cache_num_layers(cache) -> int:
    """Number of layers in the cache. Works for both APIs."""
    return len(cache)


def _get_cache_layers(cache):
    """Return the list of layers (new API) or None if legacy."""
    if _is_legacy_cache(cache):
        return None
    return getattr(cache, "layers", None) or getattr(cache, "_cache_layers", None)


def _cache_get_layer(cache, layer_idx: int):
    """Return (key_tensor, value_tensor) for the given layer index."""
    if _is_legacy_cache(cache):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    # New API: cache[i] or cache.layers[i].keys / .values
    layers = _get_cache_layers(cache)
    if layers is not None and layer_idx < len(layers):
        layer = layers[layer_idx]
        return layer.keys, layer.values
    # Fallback: __getitem__ often returns (key, value) for both APIs
    kv = cache[layer_idx]
    if isinstance(kv, (tuple, list)) and len(kv) >= 2:
        return kv[0], kv[1]
    raise AttributeError(f"Cannot get layer {layer_idx} from cache (no key_cache, no layers)")


def _cache_set_layer(cache, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
    """Set layer at index to the given key/value tensors (in-place)."""
    if _is_legacy_cache(cache):
        cache.key_cache[layer_idx] = key_states
        cache.value_cache[layer_idx] = value_states
        return
    layers = _get_cache_layers(cache)
    if layers is not None and layer_idx < len(layers):
        layers[layer_idx].keys = key_states
        layers[layer_idx].values = value_states
        return
    raise AttributeError(f"Cannot set layer {layer_idx} on cache (no key_cache, no layers)")


def _cache_append_layer(cache, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
    """Append a new layer with the given key/value tensors."""
    if _is_legacy_cache(cache):
        cache.key_cache.append(key_states)
        cache.value_cache.append(value_states)
        return
    layers = _get_cache_layers(cache)
    if layers is not None:
        try:
            from transformers.cache_utils import DynamicLayer
        except ImportError:
            raise AttributeError(
                "New-style cache has no key_cache and transformers.cache_utils.DynamicLayer not found"
            )
        layer = DynamicLayer()
        layer.keys = key_states
        layer.values = value_states
        layer.is_initialized = True
        layer.dtype = key_states.dtype
        layer.device = key_states.device
        layers.append(layer)
        return
    # New API: Cache may create layers on demand via update(key_states, value_states, layer_idx, cache_kwargs)
    update_fn = getattr(cache, "update", None)
    if callable(update_fn):
        layer_idx = len(cache)
        try:
            # Signature in new API: update(key_states, value_states, layer_idx, cache_kwargs=None)
            update_fn(key_states, value_states, layer_idx, None)
            return
        except TypeError:
            # Old DynamicCache.update(key_states, value_states, layer_idx) has no cache_kwargs
            try:
                update_fn(key_states, value_states, layer_idx)
                return
            except Exception:
                pass
        except Exception:
            pass
    raise AttributeError("Cannot append to cache (no key_cache, no layers, and update() did not work)")
