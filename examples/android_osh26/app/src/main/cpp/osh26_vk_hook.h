#pragma once
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
int osh26_vk_hook_init(void);
bool osh26_vk_hook_ready(void);
void osh26_vk_hook_free(void);
#ifdef __cplusplus
}
#endif
