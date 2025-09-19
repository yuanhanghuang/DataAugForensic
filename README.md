# DataAugForensic
Official implementation of our paper “A Forensic Framework with Diverse Data Generation for Generalizable Forgery Localization.”

# Abstract
Deep learning-based forensic techniques have emerged as the leading approach for image forgery localization. However, many existing methods struggle with overfitting to the training data, which limits their generalization performance and real-world applicability. To overcome this challenge, we propose a novel forensic framework that incorporates an advanced data augmentation technique. The framework consists of two key components: a generator and a detector. The generator challenges the detector’s learned distribution under constraints of diversity and consistency, ensuring that the generated data diverges from the source domain while maintaining statistical differences related to tampering. The detector, in turn, captures tampering traces from three critical aspects of the tampered image: long-range dependency information, RGB-noise fusion information, and boundary artifacts, resulting in a more comprehensive detection process. By alternating the optimization of the generator and detector, the framework fosters mutual reinforcement, promoting diverse data generation and expanding the distributional coverage, ultimately improving performance. Extensive experiments demonstrate that the proposed method significantly surpasses state-of-the-art approaches in both generalization and robustness, with numerous ablation studies further validating the soundness of the model design.

# Model

# Dataset
This repository provides two self-constructed forgery datasets, created with deep learning–based inpainting methods and used in our paper. They are named [IA-DO](https://pan.quark.cn/s/6dee37235207) and [PP-DO](https://pan.quark.cn/s/6dee37235207), with their real (unaltered) version referred to as [DO](https://pan.quark.cn/s/763df108d641).

Below, we provide some representative examples. As illustrated, the manipulated regions blend seamlessly with the surrounding areas, exhibiting natural transitions in color, lighting, texture, and structure. Additionally, the forged images predominantly feature intricate textured backgrounds, which significantly complicate forgery localization. ![Dataset](https://github.com/yuanhanghuang/DataAugForensic/blob/main/Images/Deep_inpainting.jpg)

# Note
If you have any question, please contact me.(huangyh375@mail2.sysu.edu.cn)
