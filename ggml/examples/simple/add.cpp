#include "TestModel.h"

#define ADD_VERSION 2

extern "C" void _mlir_ciface_add(void *a, void *b, void *res);
//extern int tvu_add();

int main() {
  constexpr std::size_t kNumElements = 64;
  TestModel</*rank*/ 1, /*inputs*/ 2, /*outputs*/ 1> model("add", ADD_VERSION,
                                                           /*verbose=*/true);

  model.initRandom<float>(kNumElements);

  // Execute the model
  _mlir_ciface_add(&model.inputs[0], &model.inputs[1], &model.outputs[0]);

  // Validate the result
  float expected[kNumElements];
  for (size_t i = 0; i < kNumElements; i++)
    expected[i] = static_cast<float *>(model.inputs[0].data)[i] +
                  static_cast<float *>(model.inputs[1].data)[i];

  return model.validateResult(0, /*expected*/ expected, /*printErrs=*/true);
  //return tvu_add();
}
