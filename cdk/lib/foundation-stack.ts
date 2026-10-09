import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import { Construct } from 'constructs';
import { ContainerBuild } from './constructs/container-build';

export interface FoundationStackProps extends cdk.StackProps {
  environment: string;
  projectName: string;
}

/**
 * Foundation Stack — always deployed regardless of mode (simple or full).
 *
 * Provides the shared resources that both Path A (SageMaker) and Path B (EKS/OSMO) need:
 * - S3 buckets for datasets, models, checkpoints
 * - ECR repositories for training and inference containers
 * - IAM roles for SageMaker
 * - CodeBuild projects that build every container image in the cloud and push
 *   to ECR — so users never pull multi-GB NVIDIA base images or run `docker
 *   build` on their laptop (it can't be done on Apple Silicon anyway).
 *
 * WORKSHOP NOTE: This is the first thing you deploy. The stack itself takes
 * ~3 minutes; the container builds it kicks off run in the background in
 * CodeBuild (groot ~10 min, isaac-lab/cosmos can take up to an hour). Watch
 * them in the CodeBuild console — see the BuildConsole output below.
 */
export class FoundationStack extends cdk.Stack {
  // Expose resources for other stacks to reference
  public readonly datasetsBucket: s3.Bucket;
  public readonly modelsBucket: s3.Bucket;
  public readonly checkpointsBucket: s3.Bucket;
  public readonly grootTrainingRepo: ecr.Repository;
  public readonly grootInferenceRepo: ecr.Repository;
  public readonly isaacLabRepo: ecr.Repository;
  public readonly isaacSimRepo: ecr.Repository;
  public readonly inferenceRepo: ecr.Repository;
  public readonly cosmosRepo: ecr.Repository;
  public readonly cosmos3Repo: ecr.Repository;
  public readonly sagemakerRole: iam.Role;

  constructor(scope: Construct, id: string, props: FoundationStackProps) {
    super(scope, id, props);

    const { projectName, environment } = props;
    const account = cdk.Aws.ACCOUNT_ID;
    const region = cdk.Aws.REGION;

    // =========================================================================
    // S3 BUCKETS
    // =========================================================================

    // Training datasets (LeRobot format for GR00T, USD for Isaac Lab)
    this.datasetsBucket = new s3.Bucket(this, 'DatasetsBucket', {
      bucketName: `${projectName}-${environment}-datasets-${account}`,
      versioned: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      removalPolicy: environment === 'prod'
        ? cdk.RemovalPolicy.RETAIN
        : cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: environment !== 'prod',
    });

    // Trained models (ONNX, TensorRT, model.tar.gz)
    this.modelsBucket = new s3.Bucket(this, 'ModelsBucket', {
      bucketName: `${projectName}-${environment}-models-${account}`,
      versioned: true, // Version for rollback
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      removalPolicy: cdk.RemovalPolicy.RETAIN, // Never auto-delete trained models
    });

    // Training checkpoints (intermediate, can be cleaned up)
    this.checkpointsBucket = new s3.Bucket(this, 'CheckpointsBucket', {
      bucketName: `${projectName}-${environment}-checkpoints-${account}`,
      versioned: false,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      lifecycleRules: [
        { expiration: cdk.Duration.days(14) }, // Auto-clean after 2 weeks
      ],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // =========================================================================
    // ECR REPOSITORIES
    // =========================================================================

    // GR00T fine-tuning container (Path A)
    this.grootTrainingRepo = new ecr.Repository(this, 'GrootTrainingRepo', {
      repositoryName: `${projectName}/groot-training`,
      imageScanOnPush: true,
      lifecycleRules: [{ maxImageCount: 5 }],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    // GR00T inference/serving container — serves a fine-tuned GR00T model as a
    // SageMaker real-time endpoint (Lab 1 deployment step).
    this.grootInferenceRepo = new ecr.Repository(this, 'GrootInferenceRepo', {
      repositoryName: `${projectName}/groot-inference`,
      imageScanOnPush: true,
      lifecycleRules: [{ maxImageCount: 5 }],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    // Isaac Lab RL training container (Path B)
    this.isaacLabRepo = new ecr.Repository(this, 'IsaacLabRepo', {
      repositoryName: `${projectName}/isaac-lab`,
      imageScanOnPush: true,
      lifecycleRules: [{ maxImageCount: 5 }],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    // Inference container — TensorRT + ROS2 (shared, used by edge)
    this.inferenceRepo = new ecr.Repository(this, 'InferenceRepo', {
      repositoryName: `${projectName}/inference`,
      imageScanOnPush: true,
      lifecycleRules: [{ maxImageCount: 10 }],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    // Isaac Sim scene-generation container (Stage 2 — Cosmos/procedural scenes)
    this.isaacSimRepo = new ecr.Repository(this, 'IsaacSimRepo', {
      repositoryName: `${projectName}/isaac-sim`,
      imageScanOnPush: true,
      lifecycleRules: [{ maxImageCount: 5 }],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    // Cosmos Transfer container — mirrored from NGC into our ECR by CodeBuild
    // (so the workstation/endpoint pulls from ECR, not a 30 GB NGC pull on the box).
    this.cosmosRepo = new ecr.Repository(this, 'CosmosRepo', {
      repositoryName: `${projectName}/cosmos-transfer`,
      imageScanOnPush: true,
      lifecycleRules: [{ maxImageCount: 3 }],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    // Cosmos 3 (cosmos-framework) — built from source by CodeBuild. The
    // actively-developed successor to the 2.5 line (unified predict/reason; transfer
    // still maturing). Weights pull from HuggingFace at runtime.
    this.cosmos3Repo = new ecr.Repository(this, 'Cosmos3Repo', {
      repositoryName: `${projectName}/cosmos3`,
      imageScanOnPush: true,
      lifecycleRules: [{ maxImageCount: 3 }],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    // =========================================================================
    // IAM: SAGEMAKER EXECUTION ROLE
    // =========================================================================

    this.sagemakerRole = new iam.Role(this, 'SageMakerExecutionRole', {
      roleName: `${projectName}-${environment}-sagemaker-role`,
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
      ],
    });

    // S3 access for all buckets
    this.datasetsBucket.grantReadWrite(this.sagemakerRole);
    this.modelsBucket.grantReadWrite(this.sagemakerRole);
    this.checkpointsBucket.grantReadWrite(this.sagemakerRole);

    // ECR pull access for training containers
    this.sagemakerRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        'ecr:GetAuthorizationToken',
        'ecr:BatchCheckLayerAvailability',
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
      ],
      resources: ['*'],
    }));

    // =========================================================================
    // IAM: COSMOS EC2 INSTANCE PROFILE
    // =========================================================================
    // The Cosmos Transfer runtime (training/scripts/cosmos_setup.py → EC2 Spot p5)
    // needs an instance profile so the box can: pull the cosmos image from ECR,
    // read the NGC/HF keys from Secrets Manager, read/write S3, and be driven over
    // SSM. cosmos_setup.py defaults to this profile name.
    const cosmosRole = new iam.Role(this, 'CosmosInstanceRole', {
      roleName: `${projectName}-${environment}-cosmos-role`,
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        // SSM Session Manager + RunShellScript (status/generate commands).
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
        // Pull the Cosmos container image from ECR.
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryReadOnly'),
      ],
    });
    // Read the NGC + HF secrets the NIM needs at startup.
    cosmosRole.addToPolicy(new iam.PolicyStatement({
      actions: ['secretsmanager:GetSecretValue'],
      resources: [
        `arn:aws:secretsmanager:${region}:${account}:secret:${projectName}/ngc-api-key*`,
        `arn:aws:secretsmanager:${region}:${account}:secret:${projectName}/hf-token*`,
        `arn:aws:secretsmanager:${region}:${account}:secret:${projectName}/nim-api-key*`,
      ],
    }));
    // S3 access for input clips / output scenes.
    this.datasetsBucket.grantReadWrite(cosmosRole);
    this.modelsBucket.grantReadWrite(cosmosRole);

    const cosmosInstanceProfile = new iam.CfnInstanceProfile(this, 'CosmosInstanceProfile', {
      instanceProfileName: `${projectName}-${environment}-cosmos-profile`,
      roles: [cosmosRole.roleName],
    });

    // =========================================================================
    // CODEBUILD: CONTAINER IMAGE BUILDS (cloud builds → ECR, no local Docker)
    // =========================================================================
    //
    // Every container image is built in CodeBuild from an S3 copy of this repo,
    // then pushed to ECR. This removes the multi-GB local `docker build`/`docker
    // pull` burden entirely — the only thing distributed in git is source.
    //
    // The repo source is uploaded to S3 ONCE (shared asset) and reused by every
    // build. Builds auto-trigger on `cdk deploy` and re-run only when the source
    // changes (the trigger keys off the asset's content hash).
    //
    // Builds that pull an NVIDIA NGC base image (isaac-lab, isaac-sim, cosmos)
    // need an NGC API key in Secrets Manager at `${projectName}/ngc-api-key`.
    // See Lab 0 for how to create it. groot-training builds from an AWS Deep
    // Learning Container base (cross-account ECR) — see requiresDlcLogin below.

    const sourceAsset = ContainerBuild.sourceAsset(this, 'ContainerSource');

    // GR00T fine-tuning container (Path A) — AWS PyTorch DLC base + Isaac-GR00T
    // from source. requiresDlcLogin grants the cross-account ECR pull from the
    // DLC account (763104351884); the buildspec also `docker login`s to it.
    const grootBuild = new ContainerBuild(this, 'GrootTrainingBuild', {
      projectName,
      imageName: 'groot-training',
      repository: this.grootTrainingRepo,
      buildSpecPath: 'containers/gr00t-training/buildspec.yml',
      sourceAsset,
      computeType: codebuild.ComputeType.LARGE,
      timeout: cdk.Duration.minutes(60),
      requiresDlcLogin: true,
    });

    // GR00T inference/serving container — same DLC base + GR00T pin as training.
    new ContainerBuild(this, 'GrootInferenceBuild', {
      projectName,
      imageName: 'groot-inference',
      repository: this.grootInferenceRepo,
      buildSpecPath: 'containers/gr00t-inference/buildspec.yml',
      sourceAsset,
      computeType: codebuild.ComputeType.LARGE,
      timeout: cdk.Duration.minutes(60),
      requiresDlcLogin: true,
    });

    // Isaac Lab RL container (Path B) — NGC base (~16 GB), needs big builder.
    new ContainerBuild(this, 'IsaacLabBuild', {
      projectName,
      imageName: 'isaac-lab',
      repository: this.isaacLabRepo,
      buildSpecPath: 'containers/isaac-lab/buildspec.yml',
      sourceAsset,
      computeType: codebuild.ComputeType.X2_LARGE, // 72 GB mem, large disk
      timeout: cdk.Duration.hours(2),
      requiresNgcLogin: true,
    });

    // Isaac Sim scene-generation container — NGC base (~15 GB).
    new ContainerBuild(this, 'IsaacSimBuild', {
      projectName,
      imageName: 'isaac-sim',
      repository: this.isaacSimRepo,
      buildSpecPath: 'containers/isaac-sim/buildspec.yml',
      sourceAsset,
      computeType: codebuild.ComputeType.X2_LARGE,
      timeout: cdk.Duration.hours(2),
      requiresNgcLogin: true,
    });

    // Inference container (edge) — TensorRT + ROS2. Two targets, both built in
    // the cloud so nothing (not even the Jetson/aarch64 image) builds locally:
    //   - x86 target  → pushed as :latest  (GPU PC / workstation testing)
    //   - jetson target → pushed as :jetson (NVIDIA Jetson, aarch64) on Graviton
    new ContainerBuild(this, 'InferenceBuild', {
      projectName,
      imageName: 'inference',
      repository: this.inferenceRepo,
      buildSpecPath: 'containers/inference/buildspec.yml',
      sourceAsset,
      computeType: codebuild.ComputeType.LARGE,
      timeout: cdk.Duration.hours(1),
    });

    // Jetson (aarch64) inference image — built on a Graviton fleet so it never
    // builds on-device. Its L4T base (l4t-tensorrt:r10.3.0-runtime) is on NGC, so
    // the build logs in with the NGC key; the tag is public-pullable with a free
    // NGC account (verified against the registry).
    new ContainerBuild(this, 'InferenceJetsonBuild', {
      projectName,
      imageName: 'inference-jetson',
      repository: this.inferenceRepo,
      buildSpecPath: 'containers/inference/buildspec.yml',
      sourceAsset,
      architecture: 'arm64', // Graviton fleet → builds the aarch64 Jetson image natively
      computeType: codebuild.ComputeType.LARGE,
      timeout: cdk.Duration.hours(1),
      requiresNgcLogin: true,
    });

    // Cosmos Transfer 2.5 — built from source (github.com/nvidia-cosmos/cosmos-transfer2.5).
    // Public CUDA base, so NO NGC key needed. Model weights are a gated HuggingFace
    // download (nvidia/Cosmos-Transfer2.5-2B) pulled at *runtime* via HF_TOKEN, not
    // here. Pin the version with COSMOS_REF.
    // NOTE: construct ID stays 'CosmosMirrorBuild' to match the already-deployed
    // logical ID (renaming it collides with the existing CodeBuild project name).
    new ContainerBuild(this, 'CosmosMirrorBuild', {
      projectName,
      imageName: 'cosmos-transfer',
      repository: this.cosmosRepo,
      buildSpecPath: 'containers/cosmos/buildspec.yml',
      sourceAsset,
      computeType: codebuild.ComputeType.X2_LARGE,
      timeout: cdk.Duration.hours(2),
      environmentVariables: {
        COSMOS_REPO: { value: 'https://github.com/nvidia-cosmos/cosmos-transfer2.5.git' },
        COSMOS_REF: { value: 'v1.5.4' }, // pin to a release tag, not the moving main branch
      },
    });

    // Cosmos 3 — built from source (github.com/NVIDIA/cosmos-framework), the
    // actively-developed successor to the 2.5 line. Public CUDA base (no NGC);
    // weights + guardrail are gated HuggingFace downloads pulled at runtime via
    // HF_TOKEN. Does text2image/video2video etc.; controlled transfer is still
    // 2.5-only (see CosmosMirrorBuild above).
    new ContainerBuild(this, 'Cosmos3Build', {
      projectName,
      imageName: 'cosmos3',
      repository: this.cosmos3Repo,
      buildSpecPath: 'containers/cosmos3/buildspec.yml',
      sourceAsset,
      computeType: codebuild.ComputeType.X2_LARGE, // big ML dep tree, needs disk
      timeout: cdk.Duration.hours(2),
      environmentVariables: {
        COSMOS3_REPO: { value: 'https://github.com/NVIDIA/cosmos-framework.git' },
        // No release tags published yet, so pin a commit: tracking main broke the build
        // when upstream moved its cu130 group to Python 3.13.
        COSMOS3_REF: { value: '0771793ad9d547c7fc3c6ff378a8d2e9b95e5d83' },
      },
    });

    // The training datasets bucket is read by builds that bake-in sample data.
    this.datasetsBucket.grantRead(grootBuild.project);

    // =========================================================================
    // OUTPUTS
    // =========================================================================

    new cdk.CfnOutput(this, 'DatasetsBucketName', {
      value: this.datasetsBucket.bucketName,
      description: 'Upload training datasets here (LeRobot format)',
      exportName: `${projectName}-${environment}-datasets-bucket`,
    });

    new cdk.CfnOutput(this, 'ModelsBucketName', {
      value: this.modelsBucket.bucketName,
      description: 'Trained models are written here',
      exportName: `${projectName}-${environment}-models-bucket`,
    });

    new cdk.CfnOutput(this, 'SageMakerRoleArn', {
      value: this.sagemakerRole.roleArn,
      description: 'Pass this as the execution role for SageMaker training jobs',
      exportName: `${projectName}-${environment}-sagemaker-role-arn`,
    });

    new cdk.CfnOutput(this, 'GrootTrainingRepoUri', {
      value: this.grootTrainingRepo.repositoryUri,
      description: 'ECR URI for the GR00T training container',
      exportName: `${projectName}-${environment}-groot-training-ecr`,
    });

    new cdk.CfnOutput(this, 'GrootInferenceRepoUri', {
      value: this.grootInferenceRepo.repositoryUri,
      description: 'ECR URI for the GR00T inference/serving container (Lab 1 deploy)',
      exportName: `${projectName}-${environment}-groot-inference-ecr`,
    });

    new cdk.CfnOutput(this, 'InferenceRepoUri', {
      value: this.inferenceRepo.repositoryUri,
      description: 'ECR URI for the inference container (edge deployment)',
      exportName: `${projectName}-${environment}-inference-ecr`,
    });

    new cdk.CfnOutput(this, 'IsaacLabRepoUri', {
      value: this.isaacLabRepo.repositoryUri,
      description: 'ECR URI for the Isaac Lab RL training container',
      exportName: `${projectName}-${environment}-isaac-lab-ecr`,
    });

    new cdk.CfnOutput(this, 'IsaacSimRepoUri', {
      value: this.isaacSimRepo.repositoryUri,
      description: 'ECR URI for the Isaac Sim scene-generation container',
      exportName: `${projectName}-${environment}-isaac-sim-ecr`,
    });

    new cdk.CfnOutput(this, 'CosmosRepoUri', {
      value: this.cosmosRepo.repositoryUri,
      description: 'ECR URI for the Cosmos Transfer 2.5 container (built from source)',
      exportName: `${projectName}-${environment}-cosmos-ecr`,
    });

    new cdk.CfnOutput(this, 'Cosmos3RepoUri', {
      value: this.cosmos3Repo.repositoryUri,
      description: 'ECR URI for the Cosmos 3 (cosmos-framework) container (built from source)',
      exportName: `${projectName}-${environment}-cosmos3-ecr`,
    });

    new cdk.CfnOutput(this, 'CosmosInstanceProfileName', {
      value: cosmosInstanceProfile.ref,
      description: 'Instance profile for the Cosmos EC2 runtime (cosmos_setup.py launch)',
      exportName: `${projectName}-${environment}-cosmos-instance-profile`,
    });

    new cdk.CfnOutput(this, 'BuildConsole', {
      value: `https://${region}.console.aws.amazon.com/codesuite/codebuild/projects?region=${region}`,
      description: 'Watch container image builds here (they run in CodeBuild after deploy)',
    });
  }
}
