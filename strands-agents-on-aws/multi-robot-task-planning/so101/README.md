# The packing cell: two SO-101 arms

The same robot host as the rest of the fleet, with a LeRobot SO-101 follower as its body and one skill, `pick`: an
Action Chunking with Transformers (ACT) policy that runs on the robot host at 30 Hz. A pick waits for the operator's
signed approval (`python fleet/approve.py arm-1`) before the arm moves, and the arm host refuses raw joint moves from
the planner, so every motion on a real arm is an approved pick.

## Run it

    pip install "lerobot[feetech]==0.6.1"
    cp so101/arms.example.json so101/arms.json      # your ports, calibration ids, camera indices and checkpoints
    ARM=arm-1 python so101/arm_host.py --preflight  # reads the motors and cameras, writes nothing
    ARM=arm-1 python so101/arm_host.py              # and ARM=arm-2 in a second terminal

Each arm's host loads its checkpoint from the Hugging Face Hub at start-up, using the `policy` in `arms.json`, so
nothing needs downloading by hand.

## The checkpoints and datasets

`arms.example.json` points at two example checkpoints and the datasets they were trained on, recorded with a leader
arm, about a hundred demonstrations per arm:

| Arm | Checkpoint | Dataset |
|---|---|---|
| arm-1 (black parts; wrist and overhead cameras) | [rsundaraws/act_arm1_fill](https://huggingface.co/rsundaraws/act_arm1_fill) | [rsundaraws/so101_arm1_fixedspot](https://huggingface.co/datasets/rsundaraws/so101_arm1_fixedspot) |
| arm-2 (blue parts; wrist camera only) | [rsundaraws/act_arm2_fill_wristonly](https://huggingface.co/rsundaraws/act_arm2_fill_wristonly) | [rsundaraws/so101_arm2_fixedspot](https://huggingface.co/datasets/rsundaraws/so101_arm2_fixedspot) |

These checkpoints pick reliably only on the setup they were recorded on: the same arm, cameras and marked spot. For
your own arms, record your own demonstrations with LeRobot, train ACT, and set `policy` in `arms.json` to your
checkpoint; `poses.json` holds the rest pose and pick ranges derived from the demonstrations.

The datasets are also what `--replay` uses: `ARM=arm-2 python so101/arm_host.py --replay 60` serves the arm with no
hardware, replaying a recorded episode, and `so101/check_without_hardware.py` then drives it over IoT Core the way the
planner and the operator would. The arm doesn't move in that test, so it checks the approval, stop, resume and
watchdog paths, not pick success.
