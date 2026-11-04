import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Asset } from 'aws-cdk-lib/aws-s3-assets';
import { Construct } from 'constructs';
import * as path from 'path';

export interface WorkstationStackProps extends cdk.StackProps {
  environment: string;
  /**
   * CIDR range allowed to connect via DCV (port 8443).
   * Use your IP (e.g., "203.0.113.1/32") or "0.0.0.0/0" for dev.
   */
  allowedCidr?: string;
  /**
   * EC2 instance type. Must be a GPU instance for Isaac Sim.
   * Default: g6e.4xlarge (1x L40S GPU, optimal for Isaac Sim).
   */
  instanceType?: string;
  /**
   * Region-to-AMI mapping for the Isaac Sim Marketplace AMI.
   * Product ID: prodview-bl35herdyozhw.
   */
  amiMapping: Record<string, string>;
  /**
   * EBS volume size in GB. Default: 512.
   */
  volumeSizeGb: number;
  /**
   * DCV remote desktop port. Default: 8443.
   */
  dcvPort: number;
  /**
   * Project name prefix used for ECR repo paths (e.g. `physical-ai/isaac-lab`).
   * Must match the FoundationStack projectName. Default: 'physical-ai'.
   */
  projectName?: string;
  /**
   * Git URL of this toolchain repo, cloned onto the workstation for iteration.
   * Defaults to the public aws-samples URL; if you deploy from a fork or a repo
   * that isn't published yet, pass `--context repoUrl=<your clone URL>` so the
   * workstation clones the right source (the clone is best-effort — a wrong URL
   * just leaves the box without the repo, it doesn't fail the deploy).
   */
  repoUrl?: string;
  /**
   * Availability Zone to pin the workstation instance to.
   * Useful for working around per-AZ GPU capacity constraints (e.g., when us-east-1a
   * has no g5.4xlarge capacity but us-east-1b does). If unset, CDK places the instance
   * deterministically in the lexically first AZ (typically *a), which may hit capacity limits.
   * Pass via --context availabilityZone=<az-id> or let the deploy-workstation.sh wrapper
   * script automatically retry across AZs on InsufficientInstanceCapacity errors.
   */
  availabilityZone?: string;
}

/**
 * Isaac Sim Development Workstation
 *
 * Deploys a GPU EC2 instance with:
 * - NVIDIA Isaac Sim Marketplace AMI (pre-installed driver + DCV + Isaac Sim)
 * - NICE DCV for remote desktop access
 * - Docker + NVIDIA Container Toolkit for Isaac Lab containers
 * - S3 access for datasets and models
 *
 * Use this to visually develop and debug Isaac Lab RL environments.
 * Connect via DCV client or web browser at https://<IP>:8443
 *
 * AMI Source: AWS Marketplace product prodview-bl35herdyozhw
 *   https://aws.amazon.com/marketplace/pp/prodview-bl35herdyozhw
 *   IMPORTANT: You MUST subscribe to this product in the AWS Marketplace before deploying.
 *   The AMI comes pre-baked with NVIDIA drivers, NICE DCV, and Isaac Sim already installed,
 *   so the bootstrap is minimal (no driver/desktop/DCV installation needed).
 *
 * Based on: https://github.com/aws-samples/sample-physical-ai-scaffolding-kit/tree/main/isaacsim-workstation
 *
 * Deploy:
 *   cdk deploy PhysicalAi-dev-Workstation --context mode=simple --context allowedCidr=YOUR_IP/32
 *
 * WORKSHOP NOTE: This is your "IDE" for robotics. Develop visually here,
 * then run headless training on SageMaker once the environment works.
 */
export class WorkstationStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: WorkstationStackProps) {
    super(scope, id, props);

    // DCV security: MUST explicitly specify allowedCidr. While DCV requires password auth,
    // the default password (pai-lab1) is documented, so 0.0.0.0/0 is not safe. Users
    // should pass --context allowedCidr=<ip>/32 or use deploy-workstation.sh which auto-detects.
    // Note: VPN often blocks port 8443 outbound — users typically connect off-VPN.
    if (!props.allowedCidr) {
      throw new Error(
        'allowedCidr is required for DCV access security. Pass --context allowedCidr=YOUR_IP/32 ' +
        'or use deploy-workstation.sh which auto-detects your IP. To explicitly allow all ' +
        '(not recommended for production), pass allowedCidr=0.0.0.0/0'
      );
    }
    const allowedCidr = props.allowedCidr;
    const instanceType = props.instanceType || 'g6e.4xlarge';
    const volumeSizeGb = props.volumeSizeGb || 512;
    const dcvPort = props.dcvPort || 8443;
    const projectName = props.projectName || 'physical-ai';
    const repoUrl = props.repoUrl || 'https://github.com/aws-samples/aws-physical-ai-toolchain.git';
    const availabilityZone = props.availabilityZone;

    // ECR registry for this account/region (resolved at deploy time via CDK tokens).
    const ecrRegistry = `${cdk.Aws.ACCOUNT_ID}.dkr.ecr.${cdk.Aws.REGION}.amazonaws.com`;
    const isaacLabImage = `${ecrRegistry}/${projectName}/isaac-lab:latest`;

    // Toolchain code delivery — bundle the local working tree as an S3 asset.
    // The public GitHub repo isn't released yet, so we can't `git clone` it on the box.
    // Instead CDK zips THIS repo at synth time, uploads it to the bootstrap assets bucket,
    // and the instance downloads + unzips it in UserData (see Step 6). Bonus: the code on
    // the workstation is exactly your local working copy. Swap to a `git clone` once the
    // repo is public (pass --context repoUrl=<public-url>).
    // Excludes keep the zip small and avoid shipping build artifacts / local CDK state.
    const toolchainAsset = new Asset(this, 'ToolchainCode', {
      path: path.join(__dirname, '../..'),
      exclude: [
        'node_modules',
        'cdk.out',
        '.git',
        '**/__pycache__',
        '*.pyc',
        'cdk.context.json',
        '.venv',
        'dist',
        'training/data', // 1.7 GB GR00T (Lab 1) dataset — not used by the Isaac workstation
      ],
    });

    // Use default VPC for simplicity
    const vpc = ec2.Vpc.fromLookup(this, 'DefaultVpc', { isDefault: true });

    // Security group: DCV (configurable port) + SSH (22)
    const sg = new ec2.SecurityGroup(this, 'WorkstationSg', {
      vpc,
      securityGroupName: `physical-ai-${props.environment}-workstation-sg`,
      description: 'Isaac Sim workstation: DCV + SSH access',
      allowAllOutbound: true,
    });
    sg.addIngressRule(ec2.Peer.ipv4(allowedCidr), ec2.Port.tcp(dcvPort), 'DCV remote desktop');
    sg.addIngressRule(ec2.Peer.ipv4(allowedCidr), ec2.Port.tcp(22), 'SSH');

    // IAM role for the instance
    const role = new iam.Role(this, 'WorkstationRole', {
      roleName: `physical-ai-${props.environment}-workstation-role`,
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });

    // S3 access for datasets and models
    role.addToPolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject', 's3:PutObject', 's3:ListBucket'],
      resources: [
        `arn:aws:s3:::physical-ai-${props.environment}-*`,
        `arn:aws:s3:::physical-ai-${props.environment}-*/*`,
      ],
    }));

    // DCV license check
    role.addToPolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject'],
      resources: ['arn:aws:s3:::dcv-license-*/*'],
    }));

    // ECR pull access (to pull our training containers for local testing)
    role.addToPolicy(new iam.PolicyStatement({
      actions: [
        'ecr:GetAuthorizationToken',
        'ecr:BatchCheckLayerAvailability',
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
      ],
      resources: ['*'],
    }));

    // Allow the instance to download the bundled toolchain code asset from the assets bucket.
    toolchainAsset.grantRead(role);

    // UserData: Minimal bootstrap for the Marketplace AMI (driver + DCV + Isaac Sim already installed)
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      '#!/bin/bash',
      'exec > /var/log/workstation-bootstrap.log 2>&1',
      'export DEBIAN_FRONTEND=noninteractive',
      '',
      'echo "=== Workstation Bootstrap Starting ==="',
      'date',
      '',
      '# Detect the default user (Marketplace AMI may use ubuntu or another account)',
      'DEFAULT_USER=$(getent passwd 1000 | cut -d: -f1 || echo ubuntu)',
      'echo "Default user: $DEFAULT_USER"',
      '',
      'echo "=== Step 1: Set user password for DCV login ==="',
      'echo "$DEFAULT_USER:pai-lab1" | chpasswd',
      '',
      'echo "=== Step 2: Install AWS CLI v2 (if missing) ==="',
      'if ! command -v aws &>/dev/null; then',
      '  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip',
      '  unzip -q /tmp/awscliv2.zip -d /tmp',
      '  /tmp/aws/install',
      '  rm -rf /tmp/aws /tmp/awscliv2.zip',
      'fi',
      '',
      'echo "=== Step 3: Install Docker + NVIDIA Container Toolkit (if missing) ==="',
      'if ! command -v docker &>/dev/null; then',
      '  curl -fsSL https://get.docker.com | sh',
      '  systemctl enable docker && systemctl start docker',
      'fi',
      'usermod -aG docker $DEFAULT_USER || true',
      '',
      'if ! dpkg -l | grep -q nvidia-container-toolkit; then',
      '  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg',
      '  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null',
      '  apt-get update -yq && apt-get install -yq nvidia-container-toolkit',
      '  systemctl restart docker',
      'fi',
      '',
      'echo "=== Step 4: Ensure DCV session exists ==="',
      'if command -v dcv &>/dev/null; then',
      '  if ! dcv list-sessions | grep -q "^Session:"; then',
      '    dcv create-session --type console --owner $DEFAULT_USER main || echo "DCV session creation failed (may already exist)"',
      '  fi',
      'fi',
      '',
      'echo "=== Step 5: ECR login + pull Isaac Lab container (best-effort) ==="',
      `aws ecr get-login-password --region ${cdk.Aws.REGION} | docker login --username AWS --password-stdin ${ecrRegistry} || echo "ECR login failed (non-fatal)"`,
      `docker pull ${isaacLabImage} || echo "isaac-lab image not in ECR yet (CodeBuild may still be running)"`,
      '',
      'echo "=== Step 6: Fetch toolchain code (S3 asset bundled at synth) ==="',
      '# The repo isn\'t public yet, so instead of git clone we download the code bundle',
      '# that CDK uploaded to the assets bucket and unzip it into the user home dir.',
      'TOOLCHAIN_DIR=/home/$DEFAULT_USER/aws-physical-ai-toolchain',
      `aws s3 cp ${toolchainAsset.s3ObjectUrl} /tmp/toolchain.zip --region ${cdk.Aws.REGION} && \\`,
      '  mkdir -p "$TOOLCHAIN_DIR" && \\',
      '  unzip -q -o /tmp/toolchain.zip -d "$TOOLCHAIN_DIR" && \\',
      '  chown -R $DEFAULT_USER:$DEFAULT_USER "$TOOLCHAIN_DIR" && \\',
      '  rm -f /tmp/toolchain.zip && \\',
      '  echo "Toolchain code unpacked to $TOOLCHAIN_DIR" || \\',
      '  echo "Toolchain code download failed (non-fatal)"',
      '',
      'echo "=== Step 6.5: Cache dirs for Isaac Sim container (faster GUI boots) ==="',
      '# Isaac Sim\'s first GUI boot syncs extensions + compiles shaders (~5 min).',
      '# Persisting these caches across container runs makes later boots much faster.',
      'sudo -u $DEFAULT_USER mkdir -p \\',
      '  /home/$DEFAULT_USER/isaac-cache/kit \\',
      '  /home/$DEFAULT_USER/isaac-cache/ov \\',
      '  /home/$DEFAULT_USER/isaac-cache/glcache \\',
      '  /home/$DEFAULT_USER/isaac-cache/computecache \\',
      '  /home/$DEFAULT_USER/isaac-cache/logs',
      '',
      'echo "=== Step 7: Create convenience scripts ==="',
      '# Single environment for ALL GPU work: the isaac-lab container (Isaac Sim 4.5.0',
      '# + Isaac Lab v2.1.0 — the SAME image SageMaker/Batch run in Lab 4, and the one',
      '# whose omni.isaac.lab.* namespace matches the repo\'s envs + eval scripts). The',
      '# Marketplace AMI\'s native Isaac Sim 5.1.0 is used only for the standalone GUI',
      '# (run-isaac-sim-gui.sh); RL train/eval go through the container so there is one',
      '# consistent environment. GUI passthrough VALIDATED on this AMI: container RTX/',
      '# Vulkan renderer initializes against the DCV display and an Anymal task trains.',
      'cat > /home/$DEFAULT_USER/run-isaac-lab.sh << RUNSCRIPT',
      '#!/bin/bash',
      '# Launch the isaac-lab container with GPU + GUI passthrough to the DCV desktop,',
      '# then drop you at a shell inside it (prompt: /workspace/isaaclab#). From there:',
      '#   Train (visual):  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \\\\',
      '#                       --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 4096',
      '#   Eval  (visual):  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \\\\',
      '#                       --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 32 \\\\',
      '#                       --checkpoint <logs/skrl/.../checkpoints/agent_<N>.pt>',
      '# Pass --headless to train.py for max throughput (no render window).',
      '# NOTE: a non-fatal "Warp CUDA error: cuDeviceGetUuid" may print on boot',
      '#       (container CUDA vs host driver) — the sim runs through it.',
      'set -e',
      `IMAGE="${isaacLabImage}"`,
      'xhost + >/dev/null 2>&1 || sudo xhost + >/dev/null 2>&1 || true',
      'CACHE="\\$HOME/isaac-cache"',
      'exec sudo docker run --rm -it \\\\',
      '  --gpus all --network=host --shm-size=16g \\\\',
      '  -e DISPLAY -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \\\\',
      '  -v /tmp/.X11-unix:/tmp/.X11-unix \\\\',
      '  -v "\\$CACHE/kit":/isaac-sim/kit/cache \\\\',
      '  -v "\\$CACHE/ov":/root/.cache/ov \\\\',
      '  -v "\\$CACHE/glcache":/root/.cache/nvidia/GLCache \\\\',
      '  -v "\\$CACHE/computecache":/root/.nv/ComputeCache \\\\',
      '  -v "\\$HOME/aws-physical-ai-toolchain":/workspace/toolchain \\\\',
      '  --entrypoint bash "\\$IMAGE" -lc "cd /workspace/isaaclab && exec bash"',
      'RUNSCRIPT',
      '',
      'cat > /home/$DEFAULT_USER/run-isaac-sim-gui.sh << \'GUISCRIPT\'',
      '#!/bin/bash',
      '# Launch Isaac Sim visual UI. The Marketplace AMI installs Isaac Sim to',
      '# /opt/IsaacSim (also mirrored under ~/IsaacSim); the launcher is NOT on PATH,',
      '# so we call it by absolute path.',
      'if [ -x /opt/IsaacSim/isaac-sim.sh ]; then',
      '  exec /opt/IsaacSim/isaac-sim.sh',
      'elif [ -x "$HOME/IsaacSim/isaac-sim.sh" ]; then',
      '  exec "$HOME/IsaacSim/isaac-sim.sh"',
      'else',
      '  echo "Isaac Sim launcher not found in /opt/IsaacSim or ~/IsaacSim" >&2',
      '  exit 1',
      'fi',
      'GUISCRIPT',
      '',
      'chmod +x /home/$DEFAULT_USER/run-isaac-lab.sh /home/$DEFAULT_USER/run-isaac-sim-gui.sh',
      'chown $DEFAULT_USER:$DEFAULT_USER /home/$DEFAULT_USER/run-isaac-lab.sh /home/$DEFAULT_USER/run-isaac-sim-gui.sh',
      '',
      'echo "Workstation bootstrap complete" > /var/log/workstation-bootstrap.summary',
      'IMDS_TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")',
      'echo "DCV URL: https://$(curl -s -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4):' + dcvPort.toString() + '"',
      'echo "Username: $DEFAULT_USER  Password: pai-lab1 (change it with: sudo passwd $DEFAULT_USER)"',
      '',
      'echo "=== Workstation Bootstrap Complete ==="',
      'date',
    );

    // EC2 Instance using the Isaac Sim Marketplace AMI
    // IMPORTANT: The AMI mapping comes from config.json and points to the NVIDIA Isaac Sim
    // Marketplace AMI (product ID prodview-bl35herdyozhw). This AMI is PRE-BAKED with:
    //   - NVIDIA GPU drivers
    //   - NICE DCV server
    //   - Isaac Sim already installed
    // You MUST subscribe to the Marketplace product before deploying, or the instance launch
    // will fail with a subscription required error. Visit:
    //   https://aws.amazon.com/marketplace/pp/prodview-bl35herdyozhw
    //
    // If availabilityZone is specified, pin the instance to that AZ (for working around
    // per-AZ GPU capacity constraints). Otherwise, CDK places it deterministically.
    const instance = new ec2.Instance(this, 'Workstation', {
      instanceType: new ec2.InstanceType(instanceType),
      machineImage: ec2.MachineImage.genericLinux(props.amiMapping),
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      securityGroup: sg,
      role,
      userData,
      requireImdsv2: true,
      blockDevices: [{
        deviceName: '/dev/sda1',
        volume: ec2.BlockDeviceVolume.ebs(volumeSizeGb, {
          volumeType: ec2.EbsDeviceVolumeType.GP3,
          encrypted: true,
        }),
      }],
      associatePublicIpAddress: true,
      ...(availabilityZone && { availabilityZone }),
    });

    // Outputs
    new cdk.CfnOutput(this, 'WorkstationInstanceId', {
      value: instance.instanceId,
      description: 'Instance ID (use for start/stop and SSM)',
    });

    new cdk.CfnOutput(this, 'DCVWebURL', {
      value: `Connect via: https://<PUBLIC_IP>:${dcvPort} (get IP from EC2 console or: aws ec2 describe-instances --instance-ids ${instance.instanceId} --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)`,
      description: 'Connect via web browser (accept cert warning)',
    });

    new cdk.CfnOutput(this, 'SSMConnect', {
      value: `aws ssm start-session --target ${instance.instanceId}`,
      description: 'Connect via Session Manager (no SSH key needed)',
    });

    new cdk.CfnOutput(this, 'SetPassword', {
      value: `aws ssm send-command --instance-ids ${instance.instanceId} --document-name "AWS-RunShellScript" --parameters 'commands=["echo ubuntu:YOUR_PASSWORD | chpasswd"]'`,
      description: 'Set the ubuntu user password for DCV login',
    });

    new cdk.CfnOutput(this, 'Cost', {
      value: `~$3.00/hr (${instanceType}, 1x L40S GPU, on-demand us-west-2) + ~$${Math.round(volumeSizeGb * 0.08)}/mo for ${volumeSizeGb}GB gp3 EBS. Actual cost varies by region. STOP instance when not in use! Start/stop via console or: aws ec2 stop-instances / start-instances`,
      description: 'Estimated hourly cost — only runs when you need it',
    });
  }
}
