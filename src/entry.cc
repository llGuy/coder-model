#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(coder_model_sim, m) 
{
    m.def("inspect", [](nb::ndarray<float, nb::shape<nb::any>> data) {
#if 0
        printf("Array data pointer : %p\n", a.data());
        printf("Array dimension : %zu\n", a.ndim());

        for (size_t i = 0; i < a.ndim(); ++i) {
            printf("Array dimension [%zu] : %zu\n", i, a.shape(i));
            printf("Array stride    [%zu] : %zd\n", i, a.stride(i));
        }

        printf("Device ID = %u (cpu=%i, cuda=%i)\n", a.device_id(),
               int(a.device_type() == nb::device::cpu::value),
               int(a.device_type() == nb::device::cuda::value));

        printf("Array dtype: int16=%i, uint32=%i, float32=%i\n",
               a.dtype() == nb::dtype<int16_t>(),
               a.dtype() == nb::dtype<uint32_t>(),
               a.dtype() == nb::dtype<float>());
#endif

        auto v = data.view();

        for (size_t i = 0; i < v.shape(0); ++i)
            v(i) = v(i) * 2;
    });
}
