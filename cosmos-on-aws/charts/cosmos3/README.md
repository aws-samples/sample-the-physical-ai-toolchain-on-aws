# Cosmos 3 Helm chart

This chart deploys one reusable Cosmos 3 server as a GPU-backed Kubernetes
`Deployment` and an internal-only `ClusterIP` `Service`. It deliberately creates
no ingress, load balancer, `ServiceAccount`, or `Secret`. Use the existing
Terraform-managed `cosmos3-generator` service account and connect through a
localhost-bound `kubectl port-forward`.

The default Nano configuration fails closed until a Region, participant-local
archive URI, and the validated archive SHA-256 are supplied.

```bash
helm upgrade --install cosmos3-g1-server ./cosmos-on-aws/charts/cosmos3 \
  --namespace default \
  --set-string aws.region="$AWS_REGION" \
  --set-string image.ref="$COSMOS_RUNTIME_IMAGE_URI" \
  --set-string model.archiveS3Uri="$COSMOS_ARCHIVE_S3_URI" \
  --set-string model.archiveSha256="$COSMOS_ARCHIVE_SHA256" \
  --wait --timeout 30m --atomic
```

Access the service only from an authorized workstation:

```bash
kubectl port-forward --address 127.0.0.1 \
  service/cosmos3-g1-server 8000:8000
```

Remove the GPU workload and service together:

```bash
helm uninstall cosmos3-g1-server --namespace default --wait --timeout 5m
```

Before installing, validate the chart with the same required values:

```bash
helm lint ./cosmos-on-aws/charts/cosmos3 \
  --set-string aws.region="$AWS_REGION" \
  --set-string image.ref="$COSMOS_RUNTIME_IMAGE_URI" \
  --set-string model.archiveS3Uri="$COSMOS_ARCHIVE_S3_URI" \
  --set-string model.archiveSha256="$COSMOS_ARCHIVE_SHA256"
```
