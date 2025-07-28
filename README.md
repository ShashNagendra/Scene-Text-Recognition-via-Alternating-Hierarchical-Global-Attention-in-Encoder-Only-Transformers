# Scene-Text-Recognition-via-Alternating-Hierarchical-Global-Attention-in-Encoder-Only-Transformers
Implementation of the paper: Scene Text Recognition via Alternating Hierarchical-Global Attention in Encoder-Only Transformers


 Install the required dependencies using: `pip install -r requirements.txt`







 **🔧 How to Run the STR/HTR Models**
 
To train or test any model, run the corresponding script from its folder:
<pre><code>```cd [folder_name] 
  ./train.ksh # For training 
  ./test.ksh # For testing ```</code></pre>



| **Folder**           | **Models**                                                                                     |
|----------------------|------------------------------------------------------------------------------------------------|
| `FasterViTSTR`          | `faster_vitstr_v0`,`faster_vitstr_v1`,`faster_vitstr_v2`,`faster_vitstr_v3`                                             |
| `DualFasterViTSTR`    | `dual_faster_vitstr_v0`,`dual_faster_vitstr_v1`,`dual_faster_vitstr_v2`,`dual_faster_vitstr_v3`                                             |



## **📦 Dataset Preparation**
This project follows the dataset structure and preparation method from the deep-text-recognition-benchmark by CLOVA AI.

**Option 1: Use Preprocessed LMDB Datasets**
Download ready-to-use LMDB datasets from the CLOVA benchmark:

📂 Directory structure:
<pre><code>``` data/
  └── data_lmdb_release/ 
  ├── training/
  └── evaluation/ ```</code></pre>
- 📎 [Download Links & Details](https://github.com/roatienza/deep-text-recognition-benchmark#download-data)


**Option 2: Create Your Own LMDB Dataset**
To use your own data, convert it to LMDB format using the `create_lmdb_dataset.py` script from the CLOVA repository:

<pre><code>```bash python3 create_lmdb_dataset.py \
  --input_path path/to/images \
  --gt_file path/to/labels.txt \ 
  --output_path data_lmdb_release/your_dataset ```</code></pre>
- 🔗 [Full instructions: CLOVA Deep Text Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

## References
We have used the following papers and their official implementations as the foundation for our models and benchmarking:
- Atienza, Rowel. "Vision transformer for fast and efficient scene text recognition." *International Conference on Document Analysis and Recognition*, pp. 319–334. Springer, 2021.
- Hatamizadeh, A., Heinrich, G., Yin, H., Tao, A., Alvarez, J. M., Kautz, J., & Molchanov, P. (2023). Fastervit: Fast vision transformers with hierarchical attention. arXiv preprint arXiv:2306.06189.




