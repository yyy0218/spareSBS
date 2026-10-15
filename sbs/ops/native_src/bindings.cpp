#include <torch/extension.h>

torch::Tensor grid_subsampling(const torch::Tensor &pc_, const float grid_size_, torch::Tensor &hash_table, torch::Tensor &hash_storage);

torch::Tensor grid_subsampling_test(const torch::Tensor &pc_, const float grid_size_, torch::Tensor &hash_table, torch::Tensor &hash_storage, uint32_t ra);

std::vector<size_t> kdtree_build(const torch::Tensor &pc, const size_t max_leaf_size);

void kdtree_free(size_t kdtree, size_t pca);

void kdtree_knn(size_t kdtree, const torch::Tensor &qpc, torch::Tensor &indices, torch::Tensor &dists, const bool sorted);

void knn_ese_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight
);

void knn_ese_backward(
    torch::Tensor &wgrad,
    const torch::Tensor &grad,
    const torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &weight
);

void knn_ese_4_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight
);

void knn_ese_4_backward(
    torch::Tensor &wgrad,
    const torch::Tensor &grad,
    const torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &weight
);

void knn_ese_4n_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight
);

void knn_ese_4n_backward(
    torch::Tensor &wgrad,
    const torch::Tensor &grad,
    const torch::Tensor &back_idx,
    const torch::Tensor &xyz,
    const torch::Tensor &weight
);



void la_ese_backward(
    torch::Tensor &f_grad,
    torch::Tensor &w_grad,
    const torch::Tensor &grad,
    const torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &xyz,
    const torch::Tensor &weight
);

void la_ese_a4_forward(
    torch::Tensor &output,
    torch::Tensor &back_idx,
    const torch::Tensor &input,
    const torch::Tensor &xyz,
    const torch::Tensor &nbr_idx,
    const torch::Tensor &weight
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_subsampling", &grid_subsampling);
    m.def("grid_subsampling_test", &grid_subsampling_test);
    m.def("kdtree_build", &kdtree_build);
    m.def("kdtree_free", &kdtree_free);
    m.def("kdtree_knn", &kdtree_knn);
    m.def("knn_ese_4_backward", &knn_ese_4_backward);
    m.def("knn_ese_4_forward", &knn_ese_4_forward);
    m.def("knn_ese_4n_backward", &knn_ese_4n_backward);
    m.def("knn_ese_4n_forward", &knn_ese_4n_forward);
    m.def("knn_ese_backward", &knn_ese_backward);
    m.def("knn_ese_forward", &knn_ese_forward);
    m.def("la_ese_a4_forward", &la_ese_a4_forward);
    m.def("la_ese_backward", &la_ese_backward);
}
