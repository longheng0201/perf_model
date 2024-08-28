# perf_model
粗略想法的一个demo
使用python编写的性能模型，用于粗略评估AI处理器的处理能力
文件说明：
（1）cfg.json：配置文件，说明cube的尺寸
（2）op.json：说明各种算子操作的种类及对应的计算公式
（3）mem_cfg.json：存储层次和容量说明
（3）parser_model.py：解析onnx
（4）test_net.py：构建测试用例，打印输出
