# Data Artifacts

The source repository intentionally does not commit the large processed runtime
artifacts under `data/processed`, `data/graph`, `data/models`, or
`data/vector_db`.

For Hugging Face Spaces, the Docker build downloads the required processed data
from this Hugging Face Dataset:

https://huggingface.co/datasets/doducdat642004/legal-qa-traffic-laws-data

Pinned deploy revision:

```text
ec787b36ad73e708a5e9615bd9ebba1c6caec7c4
```

The build hydrates these runtime paths:

- `data/processed/**`
- `data/graph/**`
- `data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder/**`
- `data/models/openvino/BAAI_bge-reranker-v2-m3/**`

These files are copied into the Docker image during build time, so they do not
appear in the Hugging Face Space **Files** tab. The **Files** tab shows only the
git repository contents. To inspect the extracted and processed data directly,
open the Dataset repository linked above.
