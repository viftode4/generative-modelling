# GenEval Balanced Prompt Set for Vlad

Purpose: a small, balanced GenEval subset for Vlad's DreamShaper-xl-v2-turbo standard-vs-Z-Sampling experiment.

Source dataset: GenEval metadata from https://github.com/djghosh13/geneval/blob/main/prompts/evaluation_metadata.jsonl

Selection rule: first 3 prompts from each GenEval tag in the source file. This gives 18 prompts total across all 6 GenEval categories.

Upstream pin: `djghosh13/geneval` at `prompts/evaluation_metadata.jsonl`. The
`source_index` field in each row records that file's line number (0-based) so
the prompt set is reconstructible without re-fetching the upstream file.

```jsonl
{"source_index": 0, "tag": "single_object", "prompt": "a photo of a bench"}
{"source_index": 1, "tag": "single_object", "prompt": "a photo of a cow"}
{"source_index": 2, "tag": "single_object", "prompt": "a photo of a bicycle"}
{"source_index": 80, "tag": "two_object", "prompt": "a photo of a bench and a sports ball"}
{"source_index": 81, "tag": "two_object", "prompt": "a photo of a toothbrush and a snowboard"}
{"source_index": 82, "tag": "two_object", "prompt": "a photo of a toaster and an oven"}
{"source_index": 179, "tag": "counting", "prompt": "a photo of two clocks"}
{"source_index": 180, "tag": "counting", "prompt": "a photo of two backpacks"}
{"source_index": 181, "tag": "counting", "prompt": "a photo of four handbags"}
{"source_index": 259, "tag": "colors", "prompt": "a photo of a blue fire hydrant"}
{"source_index": 260, "tag": "colors", "prompt": "a photo of a pink car"}
{"source_index": 261, "tag": "colors", "prompt": "a photo of a purple cup"}
{"source_index": 353, "tag": "position", "prompt": "a photo of a dog right of a teddy bear"}
{"source_index": 354, "tag": "position", "prompt": "a photo of a wine glass above a kite"}
{"source_index": 355, "tag": "position", "prompt": "a photo of a couch below a cup"}
{"source_index": 453, "tag": "color_attr", "prompt": "a photo of a purple wine glass and a black apple"}
{"source_index": 454, "tag": "color_attr", "prompt": "a photo of a green bus and a purple microwave"}
{"source_index": 455, "tag": "color_attr", "prompt": "a photo of a green skis and a brown airplane"}
```
