def get_resolution(original_resolution):
  """Takes (H,W) and returns (precrop, crop)."""
  area = original_resolution[0] * original_resolution[1]
  return (160, 128) if area < 96*96 else (512, 480)


known_dataset_sizes = {
  
  'cmnist' : (28,28),
  'waterbirds' : (224,224),
  'celebA' : (224,224)
}
def get_normalize_params(args):
    if args.model_arch == "DeiT":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else :
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    return mean, std

def get_resolution_from_dataset(dataset):
  if dataset not in known_dataset_sizes:
    raise ValueError(f"Unsupported dataset {dataset}.")
  return get_resolution(known_dataset_sizes[dataset])