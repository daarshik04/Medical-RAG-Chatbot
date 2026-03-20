FROM jenkins/jenkins:lts-jdk17

USER root

# ─────────────────────────────────────────────────────────────────────────────
# Single RUN block — installs everything and cleans up in one layer.
#
# FIX (exit code 7 = curl connection refused on raw.githubusercontent.com):
#   The build environment blocks github raw content URLs.
#   Switched trivy installation from the shell-script-via-curl approach to
#   the official Aqua Security APT repository — uses aquasecurity.github.io
#   which is typically reachable even in restricted environments.
#   Key point: the trivy apt repo uses "generic" as the suite name (not the
#   Debian codename), so it works on any Debian version without changes.
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        wget && \
    \
    # ── Docker CLI ──────────────────────────────────────────────────────────
    DPKG_ARCH=$(dpkg --print-architecture) && \
    case "$DPKG_ARCH" in \
        amd64) DOCKER_ARCH="x86_64"  ;; \
        arm64) DOCKER_ARCH="aarch64" ;; \
        armhf) DOCKER_ARCH="armhf"   ;; \
        *)     echo "Unsupported arch: $DPKG_ARCH" && exit 1 ;; \
    esac && \
    curl -fsSL "https://download.docker.com/linux/static/stable/${DOCKER_ARCH}/docker-27.4.1.tgz" \
        -o /tmp/docker.tgz && \
    tar -xzf /tmp/docker.tgz --strip-components=1 -C /usr/local/bin docker/docker && \
    rm /tmp/docker.tgz && \
    \
    # ── Trivy via official Aqua Security APT repo ───────────────────────────
    wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key \
        | gpg --dearmor \
        | tee /usr/share/keyrings/trivy.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/trivy.gpg] https://aquasecurity.github.io/trivy-repo/deb generic main" \
        | tee /etc/apt/sources.list.d/trivy.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends trivy && \
    \
    # ── Cleanup ─────────────────────────────────────────────────────────────
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    \
    # ── Sanity checks — fail the build if either binary is missing ──────────
    docker --version && \
    trivy --version

RUN groupadd -f docker && \
    usermod -aG docker jenkins

USER jenkins

# ─────────────────────────────────────────────────────────────────────────────
# Build:  docker build -f custom_jenkins.Dockerfile -t custom-jenkins .
#
# Run:    docker run -d \
#           --name jenkins \
#           -p 8080:8080 -p 50000:50000 \
#           -v jenkins_home:/var/jenkins_home \
#           -v /var/run/docker.sock:/var/run/docker.sock \
#           custom-jenkins
# ─────────────────────────────────────────────────────────────────────────────
