# DSC 680 Project 2: Milestone 1 - Project Proposal

## 1. Topic

This project, "Privacy-Preserving Federated Learning for Healthcare," aims to develop and evaluate an innovative federated learning framework that enables collaborative AI model training across healthcare institutions without sharing sensitive patient data. The core innovation lies in enabling multiple healthcare organizations to collaboratively build powerful AI models while keeping patient data local and private, addressing a critical barrier to AI adoption in healthcare.

## 2. Business Problem

Healthcare institutions worldwide are investing in artificial intelligence to improve patient outcomes, reduce costs, and enhance operational efficiency. However, these organizations face a fundamental dilemma: building effective AI models requires large, diverse datasets, but sharing patient data across institutions raises serious privacy, security, and regulatory concerns.

Traditional approaches to healthcare AI development force institutions to choose between building limited models on their own data or navigating complex data-sharing agreements that may still expose patient information. This results in siloed AI development, duplication of efforts, and models that fail to generalize across diverse patient populations.

As Rieke et al. (2020) highlight in their work on the future of digital health, the fragmentation of healthcare data across institutions creates significant challenges for developing robust AI systems. They note that "federated learning has the potential to overcome the limitations of local institutional datasets while preserving patient privacy." The healthcare sector stands to benefit enormously from this approach, as it could enable the development of AI systems trained on vastly more diverse patient populations than any single institution could access.

This project addresses this challenge by implementing federated learning, where models are trained across multiple institutions without sharing the underlying patient data. The research will explore the following key questions:

1. How can a federated learning approach enable effective collaborative AI model development while preserving patient privacy? We will quantify the trade-offs between model performance and privacy guarantees.

2. What techniques can mitigate data heterogeneity challenges across institutions with different patient populations? Healthcare data is naturally non-IID (not independently and identically distributed) across institutions, requiring specialized federated learning algorithms.

3. How do federated models compare to centralized models in terms of performance, bias, and generalizability? We will conduct rigorous comparative analyses using established healthcare metrics.

4. What privacy guarantees can be provided to ensure regulatory compliance and patient trust? We will implement and evaluate differential privacy techniques that provide mathematical guarantees against data reconstruction.

The need for this research is particularly timely, as Kaissis et al. (2020) point out that "the healthcare sector is under constant threat of data breaches, with sensitive medical information being a particularly valuable target." Their work demonstrates that federated learning can help mitigate these risks by keeping patient data within institutional boundaries while still enabling collaborative AI development.

## 3. Datasets

The project will leverage several healthcare datasets that represent the diversity of data types processed in modern healthcare settings. Access to these datasets requires appropriate credentials and ethical approvals, which will be obtained through the following processes:

### MIMIC-III (Medical Information Mart for Intensive Care)

This is a large, freely-available clinical database containing de-identified health data for over 40,000 patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.

- **Size**: Approximately 40GB uncompressed
- **Format**: 26 relational tables in CSV format
- **Key variables**: Demographics, vital signs, laboratory measurements, medications, procedures, diagnoses, clinical notes
- **Access procedure**: Available through PhysioNet after completing a required training course on human subjects research and signing a data use agreement
- **URL**: https://physionet.org/content/mimiciii/1.4/
- **Citation**: Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.

The MIMIC-III dataset has been extensively used in healthcare AI research and provides a robust foundation for critical care modeling. Johnson et al. (2016) developed this database specifically to support research in critical care medicine, making it an ideal candidate for federated learning experiments. The database includes detailed information collected during routine clinical care, including vital signs, medications, laboratory test results, procedures, and more. This heterogeneous data presents challenges and opportunities that mirror those found in real-world healthcare settings.

### PhysioNet Datasets - PTB-XL ECG Database

The PTB-XL ECG dataset is a large collection of electrocardiograms with clinical annotations.

- **Size**: Approximately 7.5GB
- **Format**: WFDB format with XML annotations
- **Key variables**: 12-lead ECG recordings, clinical diagnoses, demographic information
- **Access procedure**: Freely available through PhysioNet with proper attribution
- **URL**: https://physionet.org/content/ptb-xl/1.0.1/
- **Citation**: Wagner, P., Strodthoff, N., Bousseljot, R., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 7(1), 1-15.

The PTB-XL dataset, as described by Wagner et al. (2020), represents the largest open-access dataset of clinical 12-lead ECGs to date. Its multi-label nature and rich annotations make it particularly valuable for developing diagnostic AI models. This dataset brings the time-series modality to our federated learning framework, allowing us to explore how different data types perform under federated learning conditions.

### ISIC 2019 (International Skin Imaging Collaboration)

This dataset contains dermatoscopic images for skin lesion analysis.

- **Size**: Approximately 25GB
- **Format**: JPEG/PNG images with JSON metadata
- **Key variables**: High-resolution skin lesion images, demographic information, diagnosis labels
- **Access procedure**: Available through the ISIC Archive after registration
- **URL**: https://challenge.isic-archive.com/data/
- **Citation**: Codella, N., Rotemberg, V., Tschandl, P., Celebi, M. E., Dusza, S., Gutman, D., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., Kittler, H., & Halpern, A. (2019). Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC). arXiv:1902.03368.

The ISIC dataset represents the imaging component of our multimodal approach. Codella et al. (2019) describe this dataset as a benchmark for developing machine learning algorithms for skin lesion analysis. The imaging data presents unique challenges for federated learning, including high dimensionality and the need for substantial computational resources for processing.

For this project, these datasets will be partitioned to simulate multiple healthcare institutions with naturally occurring distribution shifts. This approach allows us to create realistic federated learning scenarios while maintaining access to the ground truth data for evaluation purposes. The project will not require obtaining new data, but rather will focus on developing methods that could be applied to real institutional data in future implementations.

Sheller et al. (2020) demonstrated the feasibility of this approach in their work on multi-institutional collaborations without sharing patient data. Their study showed that federated learning could achieve performance comparable to centralized approaches when applied to brain tumor segmentation tasks across multiple institutions. They emphasized that "data access and privacy concerns can be major roadblocks to multisite collaborations," which our approach directly addresses.

## 4. Methods

This project will implement a comprehensive federated learning approach for healthcare data with a focus on practical deployment and evaluation. The methodology encompasses several key components:

### Data Preparation and Simulation Framework

Our framework leverages parameter-efficient fine-tuning (PEFT) strategies to maximize model performance with limited healthcare data while maintaining strict privacy requirements. At its core, we utilize pre-trained medical foundation models as our base architecture, enhanced with lightweight adapter modules for efficient training. This approach enables transfer learning through progressive layer unfreezing, allowing the model to adapt to specific institutional data characteristics while preserving general medical knowledge.

To address data scarcity, we implement several efficiency-enhancing techniques. Domain-specific data augmentation will be applied to medical imaging data, while contrastive learning methods will be used to enhance representation learning. We also incorporate semi-supervised learning approaches to leverage unlabeled data and meta-learning strategies for rapid adaptation to new medical contexts.

The framework develops consistent preprocessing pipelines for each data modality (tabular, time-series, imaging) that can be deployed locally at participating institutions. For MIMIC-III data, we implement clinical variable normalization and imputation strategies following Johnson et al. (2016). ECG data preprocessing follows Wagner et al. (2020), including filtering and segmentation, while imaging data standardization adheres to Codella et al. (2019) procedures.

Our simulation environment mirrors real-world healthcare scenarios by creating non-IID data partitions across institutions, simulating varying data quantities, and modeling heterogeneous computational capabilities. This allows us to test adapter-based federated aggregation strategies under realistic conditions.

### Federated Learning Implementation

The implementation focuses on three critical aspects of federated learning in healthcare. First, the core federated learning component establishes FedAvg as our baseline algorithm, enhanced with FedProx to handle non-IID healthcare data distributions. We integrate privacy-preserving mechanisms using differential privacy to protect sensitive patient information during model training.

Privacy protection forms the second key component, implementing secure aggregation protocols to prevent data reconstruction attempts. We utilize homomorphic encryption for secure computations and carefully calibrate differential privacy parameters for healthcare data sensitivity. These measures ensure HIPAA compliance while maintaining model utility.

The third component addresses communication efficiency through model compression and quantization techniques. We implement selective parameter updates and bandwidth-efficient protocols to make the system practical for healthcare institutions with varying technical infrastructure.

### Evaluation Framework

Our evaluation approach examines three critical dimensions of the federated learning system. For model performance, we conduct comprehensive comparisons against centralized baselines, measuring accuracy, precision, and recall for specific clinical tasks. We carefully assess convergence rates and training stability across different healthcare scenarios.

Privacy and security evaluation involves quantifying privacy-utility tradeoffs and validating the system against common security attacks. We measure differential privacy guarantees to ensure patient data protection meets regulatory requirements. System efficiency assessment tracks communication costs, computational requirements, and overall scalability across diverse healthcare institutions.

### Ethical Considerations

The ethical framework prioritizes patient privacy through strict HIPAA compliance, robust differential privacy implementation, and secure data handling protocols. We address fairness by monitoring model performance across demographic groups and implementing comprehensive bias detection metrics to ensure equitable performance across institutions.

Accessibility remains a key focus, with system design accommodating institutions with limited resources. We develop tiered participation options and clear deployment documentation to enable broad adoption across the healthcare sector.

### Challenges and Mitigation

Several key challenges require careful consideration. Data heterogeneity across healthcare institutions necessitates specialized federated algorithms and adaptive aggregation strategies. We implement continuous monitoring of performance across institutions to detect and address any emerging issues.

System constraints are addressed through asynchronous training support and flexible participation options, while resource usage is optimized for various institutional capabilities. The critical balance between privacy and performance is maintained through careful calibration of privacy parameters, regular security audits, and ongoing assessment of the privacy-utility tradeoff.

## 5. References

Acar, D. A. E., Zhao, Y., Navarro, R. M., Mattina, M., Whatmough, P. N., & Saligrama, V. (2021). Federated Learning Based on Dynamic Regularization. In International Conference on Learning Representations.

Ali, M. S., Ahsan, M. M., Tasnim, L., Afrin, S., Biswas, K., Hossain, M. M., Ahmed, M. M., Hashan, R., Islam, M. K., Raman, S. (2024). Federated Learning in Healthcare: Model Misconducts, Security, Challenges, Applications, and Future Research Directions -- A Systematic Review. arXiv:2405.13832.

Horvath, A. N., Berchier, M., Nooralahzadeh, F., Allam, A., & Krauthammer, M. (2023). Exploratory Analysis of Federated Learning Methods with Differential Privacy on MIMIC-III. arXiv:2302.04208.

Hou, X., Khirirat, S., Yaqub, M., & Horvath, S. (2023). Improving Performance of Private Federated Models in Medical Image Analysis. arXiv:2304.05127.

Jiang, Y., Feng, C., Ren, J., Wei, J., Zhang, Z., Hu, Y., Liu, Y., Sun, R., Tang, X., Du, J., Wan, X., Xu, Y., Du, B., Gao, X., Wang, G., Zhou, S., Cui, S., Goh, R. S. M., Liu, Y., & Li, Z. (2024). Privacy-Preserving Federated Foundation Model for Generalist Ultrasound Artificial Intelligence. arXiv:2411.16380.

Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.

Jung, J., Jeong, H., & Huh, E. (2024). Federated Learning and RAG Integration: A Scalable Approach for Medical Large Language Models. arXiv:2412.13720.

Lin, L., Liu, Y., Wu, J., Cheng, P., Cai, Z., Wong, K. K. Y., & Tang, X. (2024). FedLPPA: Learning Personalized Prompt and Aggregation for Federated Weakly-supervised Medical Image Segmentation. arXiv:2402.17502.

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, 1273-1282.

Saha, P., Mishra, D., Wagner, F., Kamnitsas, K., & Noble, J. A. (2024). Examining Modality Incongruity in Multimodal Federated Learning for Medical Vision and Language-based Disease Detection. arXiv:2402.05294.

Saha, P., Mishra, D., Wagner, F., Kamnitsas, K., & Noble, J. A. (2024). FedPIA -- Permuting and Integrating Adapters leveraging Wasserstein Barycenters for Finetuning Foundation Models in Multi-Modal Federated Learning. arXiv:2412.14424.

Thrasher, J., Devkota, A., Siwakotai, P., Chivukula, R., Poudel, P., Hu, C., Bhattarai, B., & Gyawali, P. (2024). Multimodal Federated Learning in Healthcare: a Review. arXiv:2310.09650. 