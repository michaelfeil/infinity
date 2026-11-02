/* Copyright Infinity Core Contributors. Licensed under the repository LICENSE. */
#ifndef INFINITY_HEAD512_LAUNCH_H
#define INFINITY_HEAD512_LAUNCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Raw cold preparation for the current context.  Applies shared-memory
 * attributes to the exact six compiled kernel specializations.  No allocation,
 * device query, synchronization or GPU launch is performed. */
int32_t infinity_head512_prepare_raw_v2(void);

/* Raw warm launch bridge.  `params` points to a canonical Flash_fwd_params
 * whose layout is owned by native/torch_flash_attention/flash.h.  No context
 * query, attribute update, allocation or synchronization is performed here. */
int32_t infinity_head512_launch_raw_v2(void* params, void* stream);

#ifdef __cplusplus
}
#endif
#endif
