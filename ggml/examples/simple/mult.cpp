#include "TestModel.h"

#define MULT_VERSION 1

extern "C" void _mlir_ciface_mult(void *a, void *b, void *res);

int main() {
  constexpr std::size_t kNumElements = 32;
  TestModel</*rank*/ 1, /*inputs*/ 2, /*outputs*/ 1> model("mult", MULT_VERSION,
                                                           /*verbose=*/true);
  model.initRandom<float>(kNumElements);

  // Execute the model
  _mlir_ciface_mult(&model.inputs[0], &model.inputs[1], &model.outputs[0]);

  // Validate the result
  float expected[kNumElements];
  for (size_t i = 0; i < kNumElements; i++)
    expected[i] = static_cast<float *>(model.inputs[0].data)[i] *
                  static_cast<float *>(model.inputs[1].data)[i];

  return model.validateResult(0, /*expected*/ expected, /*printErrs=*/true);
}

