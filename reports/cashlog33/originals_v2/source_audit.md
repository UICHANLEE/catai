# CashLog33 원본 데이터 감사

- manifest: 7개
- row: 43,290개
- 고유 SHA-256: 42,995개
- 중복 hash: 294개
- cross-leaf hash 충돌: 1개

## 출처별 수량

| manifest | rows | unique hashes |
|---|---:|---:|
| `/Users/uichan/workspace/catai/data/raw/cashlog33/actual/manifest.jsonl` | 2 | 2 |
| `/Users/uichan/workspace/catai/data/raw/cashlog33/openimages_v7/manifest.jsonl` | 17,983 | 17,983 |
| `/Users/uichan/workspace/catai/data/raw/cashlog33/openverse_smoke/manifest.jsonl` | 61 | 61 |
| `/Users/uichan/workspace/catai/data/raw/cashlog33/open_products_api/manifest.jsonl` | 391 | 390 |
| `/Users/uichan/workspace/catai/data/raw/cashlog33/open_products_api_250/manifest.jsonl` | 581 | 581 |
| `/Users/uichan/workspace/catai/data/raw/cashlog33/cord_v2/manifest.jsonl` | 1,000 | 998 |
| `/Users/uichan/workspace/catai/data/raw/cashlog33/abo_v1/manifest.jsonl` | 23,272 | 23,272 |

## 주의

API·객체·상품 타입 기반 라벨은 weak label이다. 실제 CashLog 사람이 확인한
라벨과 OCR-only 영수증을 같은 정확도 근거로 합산하면 안 된다.
