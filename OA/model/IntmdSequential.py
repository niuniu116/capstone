import torch.nn as nn


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        # n = 0
        for name, module in self.named_children():
            # print(n)
            # n = n+1
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs