from pathlib import Path

import SimpleITK as sitk
import numpy as np
import sys
import torch
import nibabel as nib
from skimage.transform import resize


def iteration_callback(filter):
    global itr
    print("deformable iter:", itr, "loss:", filter.GetMetricValue(), flush=True)
    itr += 1


def save(filter, fixed, moving, fct, mct, fpathology, mpathology):
    m = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, filter,
                                             sitk.sitkLinear, 0.0,
                                             moving.GetPixelIDValue()))

    mct = resize(mct, fct.shape)
    mct = sitk.GetImageFromArray(mct, False)
    mct = sitk.GetArrayFromImage(sitk.Resample(mct, fixed, filter,
                                               sitk.sitkLinear, 0.0,
                                               mct.GetPixelIDValue()))
    if mpathology is not None:
        mpathology = resize(mpathology, fpathology.shape, order=0)
        mpathology = sitk.GetImageFromArray(mpathology, False)
        mpathology = sitk.GetArrayFromImage(sitk.Resample(mpathology, fixed, filter,
                                                          sitk.sitkLinear, 0.0,
                                                          mpathology.GetPixelIDValue()))

    return m, mct, mpathology


def log_rigid():
    global iteration
    print("rigid iter:", iteration, flush=True)
    iteration += 1


def rigid_registration(f, m):
    transform = sitk.CenteredTransformInitializer(f,
                                                  m,
                                                  sitk.Euler3DTransform(),
                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # multi-resolution rigid registration using Mutual Information
    registration_m = sitk.ImageRegistrationMethod()
    registration_m.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_m.SetMetricSamplingStrategy(registration_m.RANDOM)
    registration_m.SetMetricSamplingPercentage(0.01)
    registration_m.SetInterpolator(sitk.sitkLinear)
    registration_m.SetOptimizerAsGradientDescent(learningRate=1.0,
                                                 numberOfIterations=100,
                                                 convergenceMinimumValue=1e-6,
                                                 convergenceWindowSize=10)
    registration_m.SetOptimizerScalesFromPhysicalShift()
    registration_m.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_m.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_m.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_m.SetInitialTransform(transform)

    # add iteration callback, save central slice in xy, xz, yz planes
    global iteration_number
    iteration_number = 0
    registration_m.AddCommand(sitk.sitkIterationEvent,
                              lambda: log_rigid())

    rigid_transformation = registration_m.Execute(f, m)

    m = sitk.Resample(m, f, rigid_transformation, sitk.sitkLinear, 0.0,
                      m.GetPixelIDValue())
    print("rigid registration finished.", flush=True)
    return f, m


itr = 0
iteration = 1


def deformable_registration(fixed_image, moving_image, fixed_ct, moving_ct, fixed_pathology, moving_pathology):
    moving_image = resize(moving_image, fixed_image.shape, order=0)
    fixed_image = sitk.GetImageFromArray(fixed_image, False)
    moving_image = sitk.GetImageFromArray(moving_image, False)

    # uncommnet to do rigid registration first
    # fixed_image, moving_image = rigid_registration(fixed_image,moving_image)

    registration_method = sitk.ImageRegistrationMethod()

    # Determine the number of BSpline control points using the physical
    # spacing we want for the finest resolution control grid.
    grid_physical_spacing = [50.0, 50.0, 50.0]  # A control point every 50mm
    image_physical_size = [size * spacing for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size / grid_spacing + 0.5) \
                 for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]
    # The starting mesh size will be 1/4 of the original, it will be refined by
    # the multi-resolution framework.
    mesh_size = [int(sz / 4 + 0.5) for sz in mesh_size]

    initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
                                                         transformDomainMeshSize=mesh_size, order=3)
    # Instead of the standard SetInitialTransform we use the BSpline specific method which also
    # accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with
    # the given mesh_size at the highest pyramid level then we double it in the next lower level and
    # in the full resolution image we use a mesh that is four times the original size.
    registration_method.SetInitialTransformAsBSpline(initial_transform,
                                                     inPlace=False,
                                                     scaleFactors=[1, 2, 4])

    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)


    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,
                                                      numberOfIterations=50,
                                                      convergenceMinimumValue=1e-6,
                                                      convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: iteration_callback(registration_method))

    global itr
    itr = 0
    final_transformation = registration_method.Execute(fixed_image, moving_image)
    m, mct, mpathology = save(final_transformation, fixed_image, moving_image, fixed_ct, moving_ct, fixed_pathology,
                              moving_pathology)

    print(final_transformation, flush=True)
    print('\nOptimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return m, mct, mpathology

