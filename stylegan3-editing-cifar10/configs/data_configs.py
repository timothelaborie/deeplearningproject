from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test']
	},

	'cifar10_endocde':{
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['cifar10_train'],
		'train_target_root': dataset_paths['cifar10_train'],
		'test_source_root': dataset_paths['cifar10_test'],
		'test_target_root': dataset_paths['cifar10_test']
	},

	'cifar10_at_endocde':{
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['cifar10_at_train'],
		'train_target_root': dataset_paths['cifar10_at_train'],
		'test_source_root': dataset_paths['cifar10_at_test'],
		'test_target_root': dataset_paths['cifar10_at_test']
	}
}
