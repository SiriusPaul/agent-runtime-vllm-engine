#pragma once
#ifdef __cplusplus
extern "C" {
#endif
int osh26_vk_matmul_init(void);
int osh26_vk_register(void);
int osh26_vk_matmul(int M, int N, int K, const float *a, const float *b, float *c);
extern int g_vk_matmul_ready;
#ifdef __cplusplus
}
#endif
