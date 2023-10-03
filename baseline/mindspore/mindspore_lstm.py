
import numpy as np
import mindspore
from mindspore import nn, Tensor

import mindspore.context as context
from mindspore import Tensor, context

test_type = 2
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)
# context.set_context(mode=context.GRAPH_MODE, device_target="CPU", device_id=dev_id)
if test_type > 0:
    context.set_context(enable_graph_kernel=True)
if test_type > 1:
    # context.set_context(graph_kernel_flags="--enable_stitch_fusion=true")
    context.set_context(graph_kernel_flags="--opt_level=3 --enable_stitch_fusion=1")
    # context.set_context(graph_kernel_flags="enable_stitch_fusion")

def run_lstm():
  hidden_size, num_layer, timesteps = 256, 8, 1
  net = nn.LSTM(hidden_size, hidden_size, num_layer, has_bias=True, batch_first=True, bidirectional=False)
  input = Tensor(np.ones([1, timesteps, hidden_size]).astype(np.float32))
  h0 = Tensor(np.ones([num_layer, 1, hidden_size]).astype(np.float32))
  c0 = Tensor(np.ones([num_layer, 1, hidden_size]).astype(np.float32))
  output, (h0, c0) = net(input, (h0, c0))
  
  print(output)



if __name__=="__main__":
  run_lstm()