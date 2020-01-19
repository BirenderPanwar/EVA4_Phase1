

1. What are Channels and Kernels (according to EVA)?

Channels contains features of similar context where as Kernels are the parameters which when convolved on the inputs data creates channels.  The number of channels in the filter is the same as the number of input channels. These kernels are learned and adjusted during back-propagation.  Mostly we use odd kernel size due to its advantage of symmetry in nature to extract edges/gradients.


2. Why should we (nearly) always use 3x3 kernels?

3x3 Convolution is a powerful kernel size as we can create any higher kernel size just by combining various layers of 3X3.
3X3 convolution needs a lesser number of kernel parameters compared to when higher kernel size is used directly.
3X3 being an odd kernel size have the advantage of symmetric behavior which is useful for edge detection.
It is widely used due to it's good support for hardware accelerators.

3. How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)

199x199-->197x197-->195x195-->193x193-->191x191-->189x189 (5 Layers)

189x189-->187x187-->185x185-->183x183-->181x181-->179x179 (5 Layers)

179x179-->177x177-->175x175-->173x173-->171x171-->169x169 (5 Layers)

169x169-->167x167-->165x165-->163x163-->161x161-->159x159 (5 Layers)

159x159-->157x157-->155x155-->153x153-->151x151-->149x149 (5 Layers)

149x149-->147x147-->145x145-->143x143-->141x141-->139x139 (5 Layers)

139x139-->137x137-->135x135-->133x133-->131x131-->129x129 (5 Layers)

129x129-->127x127-->125x125-->123x123-->121x121-->119x119 (5 Layers)

119x119-->117x117-->115x115-->113x113-->111x111-->109x109 (5 Layers)

109x109-->107x107-->105x105-->103x103-->101x101-->99x99 (5 Layers)

99x99 -->97x97 -->95x95 -->93x93 -->91x91 -->89x89 (5 Layers)

89x89 -->87x87 -->85x85 -->83x83 -->81x81 -->79x79 (5 Layers)

79x79 -->77x77 -->75x75 -->73x73 -->71x71 -->69x69 (5 Layers)

69x69 -->67x67 -->65x65 -->63x63 -->61x61 -->59x59 (5 Layers)

59x59 -->57x57 -->55x55 -->53x53 -->51x51 -->49x49 (5 Layers)

49x49 -->47x47 -->45x45 -->43x43 -->41x41 -->39x39 (5 Layers)

39x39 -->37x37 -->35x35 -->33x33 -->31x31 -->29x29 (5 Layers)

29x29 -->27x27 -->25x25 -->23x23 -->21x21 -->19x19 (5 Layers)

19x19 -->17x17 -->15x15 -->13x13 -->11x11 -->9x9 (5 Layers)

9X9 -->7X7 -->5X5 -->3X3 -->1X1 (4 Layers)

Number of Layers = 19*5 + 4 = 99

As using 3X3 kernel with stride 1 reduce the input size by 2, we can also calculate the number of layers as 199/2 = 99.5.. (round to 99 layers)


4. How are kernels initialized?

Each kernels are randomly initialized in DNN and are futher learned and adjusted during back-propagation.
If required we can also initialize the kernel  such as initializing using Gaussian sampling, Sober filter etc by passing initilaizer function as layer function parameters.


5. What happens during the training of a DNN?

During training of a DNN, Filters/Kernels are learned and adjusted during back-propagation.  Loss functions are used to compare the predicted and the ground truth results and during back propagation, network adjust the kernels parameters to reduce the loss value.