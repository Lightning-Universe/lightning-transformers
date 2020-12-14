from typing import Union


def is_overridden(method_name: str, model, super_object=None) -> bool:
    assert super_object is not None
    if not hasattr(model, method_name) or not hasattr(super_object, method_name):
        # in case of calling deprecated method
        return False

    instance_attr = getattr(model, method_name)
    if not instance_attr:
        return False
    super_attr = getattr(super_object, method_name)

    # when code pointers are different, it was implemented
    if hasattr(instance_attr, 'patch_loader_code'):
        # cannot pickle __code__ so cannot verify if PatchDataloader
        # exists which shows dataloader methods have been overwritten.
        # so, we hack it by using the string representation
        is_overridden = instance_attr.patch_loader_code != str(super_attr.__code__)
    else:
        is_overridden = instance_attr.__code__ is not super_attr.__code__
    return is_overridden