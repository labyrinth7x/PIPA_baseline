# PIPA baseline
This is an simplified implmentation for the paper [Learning Deep Features via Congenerous Cosine Loss for Person Reconition](https://arxiv.org/pdf/1702.06890.pdf)<br>
It has only implemented softmax loss.<br>
The coco loss version may be released some day.<br>
## Requirement<br>
* Python 2.7
* MXNET 1.3
* numpy
* matplotlib (not necessary unless the need for the result figure)  
## Network<br>
The backbone of the network is Inception pretrained in ImageNet.<br>
You can specify the network by the param --network.
## Train & Test
Train on head
```
sh run_head.sh
```
Train on face
```
sh run_face.sh
```
Train on the whole body
```
sh run_person.sh
```
## References  
Y. Liu, H. Li, and X. Wang. Learning deep features via congenerous cosine loss for person recognition. CoRR,abs/1702.06890, 2017.
