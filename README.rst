============================
Exclusive Cross-Entropy Loss
============================

.. image:: https://img.shields.io/pypi/v/ece_loss.svg
        :alt: PyPI - Latest release
        :target: https://pypi.python.org/pypi/ece_loss

.. image:: https://img.shields.io/github/license/NKI-AI/ece_loss.svg
        :alt: GitHub - Repository license
        :target: https://github.com/NKI-AI/ece_loss/blob/main/LICENSE

A `PyTorch <http://pytorch.org/>`__ implementation of the Exclusive Cross Entropy Loss.

* Free software: `Apache 2.0 license <https://github.com/NKI-AI/ece_loss/blob/main/LICENSE>`__ (please cite our work if you use it)

Features
--------

* Perform sparse-shot learning from non-exhaustively annotated datasets
* Plug-n-play components of Binary Exclusive Cross-Entropy and Exclusive Cross-entropy as substitutes for the original Cross-entropy pytorch functions
* 1-2 changes in lines of code for Exclusive cross-entropy loss compared to native pytorch cross-entropy
* Simple and modular loss class for problem personalisation if needed
* Example training code provided for a simple segmentation case


Examples
--------

See here for the simplest example in a converged training state

.. code-block:: python

    import ExclusiveCrossEntropyLoss

    loss = ExclusiveCrossEntropyLoss()
    input = torch.randn(2, 3, requires_grad=True)
    target = torch.empty(2, dtype=torch.long).random_(3)
    output = loss(input, target)

For setting the epoch state during training the `set_epoch` must be used for the annealing functions as

.. code-block:: python

    import ExclusiveCrossEntropyLoss

    # Epoch `loss.epoch` is initialised at zero. Can be changed by `ExclusiveCrossEntropyLoss(epoch=epoch)`
    loss = ExclusiveCrossEntropyLoss()

    # Updates the current epoch. The loss value for the unallabelled samples depends heavily on the current state
    #     because the background sampling and threshold annealing function decide how much of the background class
    #     to incorporate into the loss and how strict the exclusivity condition should be.
    loss.set_epoch(100)

    input = torch.randn(2, 3, requires_grad=True)
    target = torch.empty(2, dtype=torch.long).random_(3)
    output = loss(input, target)
    # Alternatively, epoch can be given as an optional argument as `loss(input, target, epoch=epoch)`


Be default, sigmoid annealing functions are used for both the (negative) background sampling and the exclusivity condition. If a different annealing is required then this can be changed by

.. code-block:: python

    import ExclusiveCrossEntropyLoss
    annealing_funct = lambda epoch, threshold: threshold  # For no annealing
    loss = ExclusiveCrossEntropyLoss(exclusivity_threshold_annealing=annealing_funct, background_sampling_annealing=annealing_funct)


For switching off the exclusivity condition, or adjusting the other parameters of the exclusive loss this can be done by

.. code-block:: python

    import ExclusiveCrossEntropyLoss
    loss = ExclusiveCrossEntropyLoss(exclusivity_threshold=1,
                                     background_sampling_threshold=1,
                                     exclusivity_threshold_annealing=lambda epoch, threshold: threshold,
                                     background_sampling_annealing=lambda epoch, threshold: threshold,
                                     focal_loss_parameters=(1, 0),
                                    )

Lastly and very importantly, the samples that are considered as unlabelled must be defined. By default the multiclass exlusive cross entropy (ExclusiveCrossEntropyLoss) defines unlabelled samples as the ones with targets equal to zero; as defined by the `zeroth_label_as_unannotated` function.

For the binary loss (BCELoss, BCEWithLogitsLoss), this is switched off and assumes all input as unlabelled, as defined by the `_identity_mapping_single_input_torch_ones` inner function.

If custom unlabelled sample mapping is required this can be adjusted by setting the unannotated_mapping variable as

.. code-block:: python

    import ExclusiveCrossEntropyLoss
    loss = ExclusiveCrossEntropyLoss()

    loss.set_unannotated_mapping(lambda targets: targets == 1)  # For the background class being assigned integer 1


A proof of concept is provided for the TNBC dataset in the examples directory with the necessary code to use the exclusive cross-entropy loss in a segmentation task.


Install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::

    pip install -e .


Use the loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import ExclusiveCrossEntropyLoss
    loss = ExclusiveCrossEntropyLoss()
    output = loss(input, target)  # just as in the ordinary CrossEntropyLoss

For more specific usages the exclusive configuration can be adjusted by:

.. code-block:: python

    loss = ExclusiveCrossEntropyLoss(exclusivity_threshold= 0.5,
                                     background_sampling_threshold = 0.5,
                                     exclusivity_threshold_annealing = annealing_function,
                                     background_sampling_annealing = annealing_function,
                                     focal_loss_parameters = (0.2, 0.1)
                                    )  # indicating the default values and a general annealing_function

Run PyTorch Experiments
-----------------------

After installing ECE run:

.. code::

    python train_tnbc [--seed] [--lr] [--loss] [--train_path] [--train_path] [--eval_path] [--test_path] [--epochs] [--batch_size] [--device]

* Available values for ``--loss`` are ``ece`` and ``ce`` for exclusive cross-entropy and cross-entropy respectively.
* Use the ``--device`` flag to set device either ``cuda`` to train on the GPU or ``cpu`` to train on the CPU.

The simple segmentation task on TNBC on the lisa surf sara cluster, using a GTX1080-ti GPU the results are:

+------------+-----------------+---------------------------+
|   DICE     |  Cross-entropy  |  Exclusive Cross-Entropy  |
+============+=================+===========================+
| TNBC @30%  |       0.78      |            0.78           |
+------------+-----------------+---------------------------+
| TNBC @30%  |       0.08      |            0.41           |
+------------+-----------------+---------------------------+

Citation
--------

`Panteli, A., Teuwen, J., Horlings, H. and Gavves, E.; Sparse-shot Learning with Exclusive Cross-Entropy for ExtremelyMany Localisations; Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 2813-2823 <https://openaccess.thecvf.com/content/ICCV2021/html/Panteli_Sparse-Shot_Learning_With_Exclusive_Cross-Entropy_for_Extremely_Many_Localisations_ICCV_2021_paper.html>`__

If you use our code, please cite:

.. code::

    @InProceedings{Panteli_2021_ICCV,
        author    = {Panteli, Andreas and Teuwen, Jonas and Horlings, Hugo and Gavves, Efstratios},
        title     = {Sparse-Shot Learning With Exclusive Cross-Entropy for Extremely Many Localisations},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2021},
        pages     = {2813-2823}
    }
