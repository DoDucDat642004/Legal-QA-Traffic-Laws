## Summary

- 

## Validation

- [ ] `find api frontend scripts src -name '*.py' -print0 | xargs -0 python3 -m py_compile`
- [ ] `bash -n entrypoint.sh && find scripts -name '*.sh' -print0 | xargs -0 bash -n`
- [ ] `python3 -m src.data_pipeline.rag_store_sync --dry-run`

## Data And Migration Notes

- [ ] Raw PDF changes are tracked with Git LFS.
- [ ] README and requirements are updated when runtime behavior changes.
- [ ] `.env.example` files are updated when new environment variables are introduced.
- [ ] Generated local outputs are not committed.

## Review Focus

- 
