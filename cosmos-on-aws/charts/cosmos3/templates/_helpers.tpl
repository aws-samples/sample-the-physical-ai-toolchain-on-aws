{{- define "cosmos3.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "cosmos3.fullname" -}}
{{- default .Release.Name .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "cosmos3.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | quote }}
app.kubernetes.io/name: {{ include "cosmos3.name" . | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service | quote }}
{{- end -}}

{{- define "cosmos3.selectorLabels" -}}
app: {{ include "cosmos3.fullname" . | quote }}
app.kubernetes.io/name: {{ include "cosmos3.name" . | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
{{- end -}}
