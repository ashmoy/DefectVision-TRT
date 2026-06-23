CATEGORY ?= bottle
CONFIG ?= configs/default.yaml

setup:
	python -m pip install -U pip
	python -m pip install -r requirements.txt
	python -m pip install -e .

train:
	python -m defectvision.train_patchcore --config $(CONFIG) --category $(CATEGORY)

eval:
	python -m defectvision.eval_patchcore --config $(CONFIG) --category $(CATEGORY)

export:
	python -m defectvision.export_onnx --config $(CONFIG) --category $(CATEGORY)

benchmark:
	python -m defectvision.benchmark --config $(CONFIG) --category $(CATEGORY)

trt:
	bash scripts/build_tensorrt_engine.sh $(CATEGORY) 256
