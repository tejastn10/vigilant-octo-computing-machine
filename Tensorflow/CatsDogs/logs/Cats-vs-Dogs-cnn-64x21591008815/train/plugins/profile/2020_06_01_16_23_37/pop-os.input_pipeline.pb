	+3����L@+3����L@!+3����L@	��/��?��/��?!��/��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$+3����L@��h8e�?A̸���L@Y�I�_{�?*	��Q�.Z@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�{/�h�?!y�1�I=@)���A��?1���q�:@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�3w�ɟ?!ОG^�=@)���ދ/�?1c��B�j8@:Preprocessing2F
Iterator::ModelP�i4��?!�u�0?@)2��8*7�?15�4~0@:Preprocessing2S
Iterator::Model::ParallelMapo�o�>;�?!���-E.@)o�o�>;�?1���-E.@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip����r�?!��b��3Q@)=e5]Ot}?1�
�Sw@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��8a�hv?!�PmH�@)��8a�hv?1�PmH�@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap"��<��?!�\��AT@@)��� �i?1ظe#@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensorkH�c�Cg?!�m>��@)kH�c�Cg?1�m>��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��h8e�?��h8e�?!��h8e�?      ��!       "      ��!       *      ��!       2	̸���L@̸���L@!̸���L@:      ��!       B      ��!       J	�I�_{�?�I�_{�?!�I�_{�?R      ��!       Z	�I�_{�?�I�_{�?!�I�_{�?JCPU_ONLY