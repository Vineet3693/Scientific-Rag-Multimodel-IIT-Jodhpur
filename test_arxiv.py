from pipelines.offline_pipeline import ArxivDataset
ds = ArxivDataset(query='vision transformer', category='cs.CV', max_results=30, keep_best=10, output_dir='data/raw/')
results = ds.download()
print("\n" + "="*80)
print("DOWNLOAD RESULTS WITH PAPERS TITLES:")
print("="*80)
for r in results:
    print(f'[{r["status"].upper()}] {r["arxiv_id"]} - "{r["title"]}"')
print("="*80)
