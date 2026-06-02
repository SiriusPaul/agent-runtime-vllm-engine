/* ggml registration for osh26_vk_matmul GPU backend */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "osh26_vk_matmul.h"

extern "C" {
extern void register_backend(ggml_backend_reg_t reg);
}

static ggml_backend_dev_t g_dev_ptr;

int osh26_vk_register(void) {
    if (!g_vk_matmul_ready) return -1;

    static struct dev_ctx { } ctx;
    static ggml_backend_device_i dev_i {};

    dev_i.get_name   = [](ggml_backend_dev_t) -> const char* { return "OSH26_Vulkan"; };
    dev_i.get_description = [](ggml_backend_dev_t) -> const char* { return "Minimal GPU"; };
    dev_i.get_memory = [](ggml_backend_dev_t, size_t *f, size_t *t) { *f = *t = 1024*1024*1024; };
    dev_i.get_type   = [](ggml_backend_dev_t) -> ggml_backend_dev_type { return GGML_BACKEND_DEVICE_TYPE_GPU; };
    dev_i.get_props  = [](ggml_backend_dev_t, ggml_backend_dev_props *p) {
        p->name = "OSH26 Vulkan"; p->description = "GPU"; p->type = GGML_BACKEND_DEVICE_TYPE_GPU;
        p->memory_free = p->memory_total = 1024*1024*1024;
        p->caps = ggml_backend_dev_caps{false, false, false, false};
    };
    dev_i.supports_op = [](ggml_backend_dev_t, const ggml_tensor *op) -> bool {
        return op->op == GGML_OP_MUL_MAT;
    };
    dev_i.offload_op = [](ggml_backend_dev_t, const ggml_tensor *op) -> bool {
        return op->op == GGML_OP_MUL_MAT;
    };
    dev_i.supports_buft = nullptr; /* accept any buffer type */

    /* init_backend */
    struct backend_data { ggml_backend_t b; ggml_backend_dev_t dev; };
    dev_i.init_backend = [](ggml_backend_dev_t dev, const char *desc) -> ggml_backend_t {
        auto *d = new backend_data;
        d->dev = dev;
        static ggml_backend_i iface;
        iface.graph_compute = [](ggml_backend_t b, ggml_cgraph *cg) -> ggml_status {
            for (int i = 0; i < cg->n_nodes; i++) {
                ggml_tensor *t = cg->nodes[i];
                if (t->op != GGML_OP_MUL_MAT) continue;
                ggml_tensor *a = t->src[0], *b_ = t->src[1];
                int64_t M = t->ne[1], N = t->ne[0], K = a->ne[0];
                if (M <= 0 || N <= 0 || K <= 0) continue;
                osh26_vk_matmul((int)M, (int)N, (int)K,
                    (const float *)a->data, (const float *)b_->data, (float *)t->data);
            }
            return GGML_STATUS_SUCCESS;
        };
        iface.free_backend = [](ggml_backend_t b) { delete ((backend_data*)b->context); };
        d->b.iface = &iface;
        d->b.device = dev;
        d->b.context = d;
        return &d->b;
        (void)desc;
    };

    /* buffer type - simple non-buffer (CPU data direct) */
    dev_i.get_buffer_type = [](ggml_backend_dev_t d) -> ggml_backend_buffer_type_t {
        static ggml_backend_buffer_type_i i;
        static ggml_backend_buffer_type bt;
        i.get_name = [](ggml_backend_buffer_type_t) -> const char* { return "VK_SIMPLE"; };
        i.alloc_buffer = [](ggml_backend_buffer_type_t, size_t sz) -> ggml_backend_buffer_t {
            void *data = calloc(1, sz);
            return ggml_backend_buffer_t{0, nullptr, data, sz, nullptr};
        };
        i.free_buffer = [](ggml_backend_buffer_t b) { free(b.context); };
        i.get_alignment = [](ggml_backend_buffer_type_t) -> size_t { return 256; };
        i.is_host = [](ggml_backend_buffer_type_t) -> bool { return true; };
        bt.iface = &i; bt.context = &bt;
        return &bt;
    };
    dev_i.get_host_buffer_type = nullptr;
    dev_i.buffer_from_host_ptr = nullptr;
    dev_i.event_new = nullptr; dev_i.event_free = nullptr;
    dev_i.event_record = nullptr; dev_i.event_wait = nullptr; dev_i.event_synchronize = nullptr;

    g_dev_ptr = new ggml_backend_device;
    g_dev_ptr->api_version = GGML_BACKEND_API_VERSION;
    g_dev_ptr->iface = &dev_i;
    g_dev_ptr->context = &ctx;

    static ggml_backend_dev_t *dev_list[] = {g_dev_ptr, nullptr};
    static ggml_backend_reg reg {};
    reg.api_version = GGML_BACKEND_API_VERSION;
    reg.iface = nullptr;
    reg.context = nullptr;
    reg.devices = dev_list;
    register_backend(&reg);

    ggml_backend_load_all();
    return 0;
}
