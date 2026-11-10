# Individual Reflection - Ameya Kamble (AIML-02)

**What I owned.** I took the project from the brief to a working, evaluated system. I ran the full pipeline on a
Colab GPU, checked every result, ran the Orthanc DICOM server, exercised the React viewer and its audit log, and
assembled the submission. I worked with an AI assistant (Claude) to write the code, and I was responsible for running it,
reading the outputs critically and deciding what could honestly be claimed.

**What I learned.**
- *DICOM is more than an image.* A file carries patient/study/series identifiers and pixel data, and a server such as
  Orthanc lets a service pull new studies and write results back, which is how clinical AI is wired in practice.
- *Transfer learning works, but limits show up fast.* A DenseNet pre-trained on ImageNet reached a mean AUROC of about
  0.80 on 14 chest findings after 4 epochs, yet some findings (infiltration, nodule) stayed weak at low image resolution.
- *Explanations need scepticism.* Grad-CAM helps a reviewer, but in several examples the heatmap pointed at a text
  label or marker rather than the lungs. That showed me why models should be reviewed by humans and why bias and shortcut
  learning must be documented.
- *Auditability is a feature.* A hash-chained, append-only audit log of each decision (with the AI output the reviewer
  saw) makes review traceable and tamper-evident.
- *Segmentation is a different problem.* A 3-D U-Net on the hippocampus reached a mean Dice of about 0.87 on held-out volumes.

**What I'd improve with more time.** Use full-resolution images (e.g. CheXpert) and external validation, measure
calibration and performance by age/sex subgroup, fine-tune on patient-level splits, add a proper React build and
authentication to the viewer, run the Docker Compose stack end to end (it is written but I only ran Orthanc and the
service directly on Colab), and wire the segmentation model into the DICOM service as well.
