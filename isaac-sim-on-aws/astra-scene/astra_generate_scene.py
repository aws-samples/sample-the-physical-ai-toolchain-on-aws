#!/usr/bin/env python3
"""Ask OpenAI GPT-6 Astra on Amazon Bedrock for a Blender scene script.

Sends a reference image (and, on a refinement pass, the previous script and its
rendered preview) to the Bedrock Converse API through the AWS CLI, then saves the
Python code Astra returns as a Blender script. Standard library only, so it runs
with the workstation's system python3 and the instance role's credentials.

    python3 astra_generate_scene.py --image reference.png --out create_scene.py
    python3 astra_generate_scene.py --image reference.png --out create_scene_v2.py \
        --previous create_scene.py --preview preview.png
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile

DEFAULT_MODEL = "us.openai.gpt-6-astra"

SYSTEM = (
    "You write Blender 4.5 Python (bpy) scripts that build simple, simulation-ready "
    "3D scenes from a reference photo. Reply with exactly one ```python code block "
    "and nothing else."
)

TASK = """Recreate the tabletop scene in the attached image as a Blender scene.

The image comes from a robot's head camera looking down at a table. Model only what
a robot simulator needs:
- a floor plane at z = 0;
- one rectangular table (top plus four legs) whose top surface is at z = {table_z} m;
- the blue box on the table, as a single cube object named exactly "Box", size about
  {box_size} m (x depth, y width, z height), resting on the table top, with ONE solid
  blue material on every face (no bands, labels or second colour);
- any other clearly visible large objects as simple boxes or cylinders;
- one sun light and one area light, and one camera named "PreviewCamera" that looks
  at the table from the front and slightly above, framing the whole table.

Do NOT model the robot, its arms, hands or fingers, even though they are in the photo:
the simulator adds the robot itself.

Rules:
- Start by deleting every default object. Use meters and Z-up.
- Name the table top "TableTop" and the legs "TableLeg_1" to "TableLeg_4".
- Put the table centre at x = {table_x} m, y = 0. The robot stands at the origin
  facing +x, so the near table edge faces the origin.
- Give each object a simple Principled BSDF material whose base colour matches the
  image. Do not use image textures or external files.
- Apply all scale and rotation transforms before export.
- Read the output folder from the environment variable SCENE_OUT_DIR.
- Save "scene.blend", export the whole scene with
  bpy.ops.wm.usd_export(filepath=..., export_materials=True) to "scene.usdc", and
  render a 640x480 Eevee image from PreviewCamera to "preview.png", all inside
  SCENE_OUT_DIR. Pick the Eevee engine name that exists in this Blender build:
  "BLENDER_EEVEE_NEXT" if it is in
  bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys(), otherwise
  "BLENDER_EEVEE".
- Do not import any module other than bpy, math, mathutils and os.
- Print "SCENE_DONE" with the object count as the last line.
"""

REFINE = """Here is the script you wrote last time and the preview it rendered (second
image). Compare the preview with the reference photo (first image). Fix wrong colours,
proportions, missing large objects or bad framing, keep every rule above (in
particular: no robot parts, and the box stays one solid blue), and return the
complete corrected script."""


def image_block(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    fmt = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}.get(ext)
    if not fmt:
        sys.exit(f"FAIL - unsupported image type: {path}")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return {"image": {"format": fmt, "source": {"bytes": data}}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True, help="reference photo of the scene")
    ap.add_argument("--out", required=True, help="where to save the Blender script")
    ap.add_argument("--previous", help="script from the last pass (refinement)")
    ap.add_argument("--preview", help="preview.png rendered by --previous (refinement)")
    ap.add_argument("--model", default=os.environ.get("ASTRA_MODEL_ID", DEFAULT_MODEL))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-2"))
    ap.add_argument("--table-z", type=float, default=0.67)
    ap.add_argument("--table-x", type=float, default=0.48)
    ap.add_argument("--box-size", default="0.08 x 0.14 x 0.09")
    ap.add_argument("--max-tokens", type=int, default=8000)
    args = ap.parse_args()

    if bool(args.previous) != bool(args.preview):
        sys.exit("FAIL - pass --previous and --preview together")

    content = [image_block(args.image),
               {"text": TASK.format(table_z=args.table_z, table_x=args.table_x,
                                    box_size=args.box_size)}]
    if args.previous:
        with open(args.previous) as f:
            content += [image_block(args.preview),
                        {"text": REFINE + "\n\n```python\n" + f.read() + "\n```"}]

    with tempfile.TemporaryDirectory() as tmp:
        msg_path = os.path.join(tmp, "messages.json")
        sys_path = os.path.join(tmp, "system.json")
        with open(msg_path, "w") as f:
            json.dump([{"role": "user", "content": content}], f)
        with open(sys_path, "w") as f:
            json.dump([{"text": SYSTEM}], f)
        cmd = ["aws", "bedrock-runtime", "converse", "--region", args.region,
               "--model-id", args.model,
               "--system", "file://" + sys_path,
               "--messages", "file://" + msg_path,
               "--inference-config", json.dumps({"maxTokens": args.max_tokens}),
               "--cli-read-timeout", "600", "--output", "json"]
        res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("FAIL - Bedrock call failed:\n" + res.stderr.strip())

    reply = json.loads(res.stdout)
    text = "".join(c.get("text", "") for c in reply["output"]["message"]["content"])
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = (match.group(1) if match else text).strip() + "\n"
    if "import bpy" not in code:
        sys.exit("FAIL - the reply did not contain a Blender script")

    with open(args.out, "w") as f:
        f.write(code)
    usage = reply.get("usage", {})
    print(f"PASS - saved {args.out} ({len(code.splitlines())} lines, "
          f"{usage.get('inputTokens', '?')} input / {usage.get('outputTokens', '?')} output tokens)")


if __name__ == "__main__":
    main()
