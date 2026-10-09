# Running the labs on AWS (optional)

Google Colab is the default runtime for every lab (PLAN.md §6), and most labs are designed for Colab's CPU runtime. This directory is an **optional** path for people who want their own NVIDIA GPU on AWS:

- **Option A:** one EC2 GPU instance per participant, created with CloudFormation and reached through SSM port forwarding. No inbound ports are open.
- **Option B:** a SageMaker AI Studio JupyterLab space, which shuts itself down when idle.

> **Status.** Nothing here has been run against AWS yet. The templates and scripts are linted, and the wrapper script was exercised offline against a fake `aws` command. A full human run is a release-checklist item. See [Unverified](#unverified).

| File | What it is |
|---|---|
| `instance.yaml` | CloudFormation template for one participant instance |
| `account-budget.yaml` | One-time, account-wide monthly cost budget with email alerts |
| `bin/prl-aws` | Wrapper for the AWS CLI: `create`, `connect`, `stop`, `start`, `status`, `destroy`, `cohort-up`, `cohort-down`, `budget` |
| `price_table.py` | Reads on-demand prices from the public AWS Price List and writes `prices.json` and the table below |
| `prices.json` | The prices from the last run, with date, offer version and source URL |
| `lint.sh` | `cfn-lint` on the templates; `shellcheck` on the scripts and on the instance's user-data |

## Costs

<!-- BEGIN generated prices (python infra/aws/price_table.py; do not edit by hand) -->

On-demand list prices in **us-east-1**, checked **2026-10-09** (UTC) by `price_table.py`. They exclude tax, data transfer, and any discounts or credits.

| Option | Instance type | GPUs | vCPU | Memory | USD per hour | 40 running hours |
|---|---|---|---|---|---|---|
| EC2 (Option A) | `g4dn.xlarge` | 1 | 4 | 16 GiB | 0.526 | 21.04 |
| EC2 (Option A) | `g5.xlarge` | 1 | 4 | 16 GiB | 1.006 | 40.24 |
| EC2 (Option A) | `g6.xlarge` | 1 | 4 | 16 GiB | 0.8048 | 32.19 |
| SageMaker Studio JupyterLab space (Option B) | `ml.g4dn.xlarge` | 1 | 4 | 16 GiB | 0.7364 | 29.46 |
| SageMaker notebook instance | `ml.g4dn.xlarge` | 1 | 4 | 16 GiB | 0.736 | 29.44 |

EBS gp3 storage: 0.08 USD per GB-month, so the default 100 GiB volume costs about 8.00 USD per month, billed while the volume exists (running or stopped) and prorated.

The last column is the hourly price × 40 hours (5 days × 8 hours), an illustration rather than a measurement.

Sources:
- EC2: <https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/20261008184850/us-east-1/index.csv> (publication date 2026-10-08T18:48:50Z, version 20261008184850)
- SageMaker: <https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/20261007160901/us-east-1/index.csv> (publication date 2026-10-07T16:09:01Z, version 20261007160901)

<!-- END generated prices -->

What else to know about cost:

- **A stopped instance** costs only its EBS storage. Each start bills at least one minute of instance time ([EC2 stop and start][ec2-stop-start]).
- **A public IPv4 address** is billed per hour while the instance runs, and the default VPC gives the instance one ([Amazon VPC pricing][vpc-pricing]). The table does not include it.
- **Data transfer** and taxes are not included.
- **`destroy` and `cohort-down`** remove the disk too. After that, the participant's stack costs nothing.

Regenerate the numbers with `python infra/aws/price_table.py` (add `--region` for another region). The script needs no credentials. It streams the EC2 offer file, about 290 MiB for us-east-1, and saves only `prices.json`. The run on 2026-10-09 took under two minutes. Run it by hand or from a scheduled job, never at site render. Never edit prices by hand.

## Prerequisites

1. **An AWS account**, and someone who can create CloudFormation stacks with IAM roles (`CAPABILITY_IAM`), EC2 instances, security groups and CloudWatch alarms.
2. **AWS CLI v2** ([install][cli-install]). `aws --version` must print `aws-cli/2…`. If it fails with "exec format error", the installed binary is for another CPU architecture; reinstall it.
3. **The Session Manager plugin** for the AWS CLI, for `connect` ([install][plugin-install]). AWS asks for version 1.2.764.0 or later (`session-manager-plugin --version`).
4. **Credentials from your own AWS CLI configuration.** IAM Identity Center works well: `aws configure sso`, then `aws sso login --profile <name>`. Pass `--profile <name>` to `prl-aws`, or set `AWS_PROFILE`. Never commit credentials. Nothing in this directory reads or writes credential files.
5. **GPU quota.** New accounts usually cannot launch any G instance. The EC2 quota **"Running On-Demand G and VT instances"** (quota code `L-DB2E81BA`) is counted in vCPUs, and AWS lists its default as 0 ([EC2 instance quotas][ec2-quotas]). Every instance type here (g4dn.xlarge, g5.xlarge, g6.xlarge) has 4 vCPUs. So you need **4 vCPUs per participant**, plus 4 for a facilitator test instance. Request the increase well before the workshop. Larger requests go to AWS Support and take time ([requesting a quota increase][quota-increase]).
   ```sh
   aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region us-east-1
   aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA \
       --desired-value 48 --region us-east-1     # for example, 12 participants x 4 vCPUs
   ```
   You can also use the console: Service Quotas → AWS services → Amazon EC2 → Running On-Demand G and VT instances → Request increase at account level.
6. **The service-linked role `AWSServiceRoleForCloudWatchEvents`.** The idle-stop alarm needs it ([CloudWatch alarm actions][alarm-actions]). `prl-aws create` and `cohort-up` create it if it is missing, which needs `iam:CreateServiceLinkedRole`. To create it yourself: `aws iam create-service-linked-role --aws-service-name events.amazonaws.com`.
7. **A region.** The working default is `us-east-1` (OPEN_QUESTIONS C2). Use the same region for every command; `--region` or `AWS_REGION` sets it.

## Option A: one EC2 instance per participant

### What a stack contains

Each participant gets one stack, `prl-<cohort>-<participant>`, built from `instance.yaml`:

| Resource | Details |
|---|---|
| EC2 instance | `g4dn.xlarge` by default (NVIDIA T4). `g5.xlarge` (A10G) and `g6.xlarge` (L4) are allowed ([accelerated computing specifications][ac-specs]). The AMI is the **Base OSS Nvidia Driver GPU DLAMI (Ubuntu 24.04)**. IMDSv2 is required. |
| AMI ID | `prl-aws` resolves it **once** from the public SSM parameter `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id` and passes the literal ID. A stack update therefore never swaps the AMI, which would replace the instance and lose the participant's files. |
| Root volume | gp3, 100 GiB by default, encrypted, `DeleteOnTermination: true` |
| Security group | **No inbound rules.** It keeps the default allow-all outbound rule. |
| IAM role | `AmazonSSMManagedInstanceCore`, plus an inline policy that allows `ssm:PutParameter` and `ssm:GetParameter` only on `parameter/prl/<stack>/*` |
| CloudWatch alarm | Stops the instance after `IdleMinutes` (default 90) of low CPU (see [Idle auto-stop](#idle-auto-stop)) |
| Tags | `Project=practical-rl`, `Cohort`, `Participant` and `Owner` on the stack, the instance, its root volume, the security group, the role and the alarm. CloudFormation cannot tag the IAM instance profile; `cohort-down` finds it by name instead. |

At first boot, the instance's user-data does the following (log: `/var/log/prl-setup.log`):

1. Installs AWS CLI v2 if the AMI lacks it.
2. Installs uv 0.12.24, then runs `uv python install 3.13`. The AMI's system Python is 3.12.
3. Clones the repository at `RepoRef` (default `main`) into `/home/ubuntu/practical-rl`.
4. Runs `uv sync --group notebooks`, with `--locked` when `uv.lock` exists. If the project has no `notebooks` group, or no `pyproject.toml` yet, it logs a warning and makes sure JupyterLab is installed in `.venv`.
5. Generates a random Jupyter token and stores it as the SSM SecureString `/prl/<stack>/jupyter-token`.
6. Enables the systemd unit `prl-jupyter.service`. It runs JupyterLab as `ubuntu` on `127.0.0.1:8888` with `PRL_PLATFORM=aws`, and `Restart=always`.
7. Enables `prl-idle-rearm.service`, which runs at every boot (see [Idle auto-stop](#idle-auto-stop)).

Progress goes to the SSM parameter `/prl/<stack>/setup-status` (`running`, `complete` or `failed …`).

### Create, connect, use, stop, start, destroy

```sh
infra/aws/bin/prl-aws create ravi --cohort oct26 --region us-east-1        # prints the plan, then waits
infra/aws/bin/prl-aws connect ravi --cohort oct26 --region us-east-1       # keep this terminal open
```

`create` resolves the AMI, prints what it will create, creates the stack and waits for `CREATE_COMPLETE`. Setup then continues on the instance for several minutes (not yet measured). The stack's `ConnectCommand` and `TokenCommand` outputs show the raw AWS CLI commands if you prefer them.

`connect` does four things:

- checks that the Session Manager plugin is installed (if it isn't, it prints the install link and stops);
- checks that the instance is running and setup is complete;
- prints `http://localhost:8888/lab?token=…`;
- forwards local port 8888 to the instance with `AWS-StartPortForwardingSession`.

Open the URL in your browser and press Ctrl+C in the terminal when you are done. If port 8888 is busy on your machine, add `--port 8899`.

**Use.** Notebooks live in `/home/ubuntu/practical-rl`. Kernels see `PRL_PLATFORM=aws`. Check the GPU from a JupyterLab terminal with `nvidia-smi`.

**Stop and start.** Stop the instance whenever you finish for the day:

```sh
infra/aws/bin/prl-aws stop ravi --cohort oct26
infra/aws/bin/prl-aws start ravi --cohort oct26     # JupyterLab restarts by itself; then connect again
infra/aws/bin/prl-aws status ravi --cohort oct26    # stack, instance state, idle alarm, setup status
```

What survives a stop and start ([EC2 stop and start][ec2-stop-start]):

- **Kept:** the EBS root volume and everything on it. That includes `/home/ubuntu/practical-rl` (your edits, saved notebooks and `.venv`), the uv cache, the token and the systemd units.
- **Lost:** memory, so every running kernel and its variables, and the instance-store NVMe disk. The DLAMI mounts it at `/opt/dlami/nvme` (DLAMI release notes); do not keep work there. The public IP address also changes, which does not matter for SSM.

**Destroy** when the participant is finished. This deletes the instance, **its disk and every file on it**, the security group, the role, the alarm, and the SSM parameters under `/prl/<stack>/`. It asks for confirmation unless you pass `--yes`.

```sh
infra/aws/bin/prl-aws destroy ravi --cohort oct26
```

**Updating a stack.** `prl-aws create <participant> --update` applies the current template to an existing stack. It can also apply `--type` and `--idle-minutes`, and it reuses the stack's AMI ID. A new instance type or new user-data stops and restarts the instance, and the new user-data does **not** re-run (cloud-init runs it once per instance). The volume size cannot change this way.

### Idle auto-stop

The CloudWatch alarm watches `CPUUtilization` in 5-minute periods. It stops the instance when **every** 5-minute average stays below 5% for `IdleMinutes` minutes; the default is 90, so it spans a lunch break. The action is `arn:aws:automate:<region>:ec2:stop`. A cell that is training typically keeps one of the four vCPUs busy, which is about 25% on this metric. Reading and editing in JupyterLab counts as idle.

- **Missing data.** The alarm treats missing data as `missing`, as AWS recommends for alarms that stop instances ([missing data][missing-data]).
- **Re-arming after a restart.** EC2 alarm actions run only when the alarm changes state, not again while it stays in a state ([alarm actions][alarm-actions-overview]). If the alarm stopped the instance and someone starts it again soon after, the alarm can still be in `ALARM`, and an idle instance would then never be stopped. So every boot runs `prl-idle-rearm.service`, which keeps one vCPU busy at the lowest priority for 6 minutes. That moves the alarm back to `OK`, so it can fire again.
- **Choosing the delay.** `--idle-minutes` takes 30, 60, 90, 120, 180 or 240.

Idle stop is a safety net, not a substitute for `stop` and `destroy`.

### Network and egress

With no `--subnet`, the instance goes into a default subnet of the account's **default VPC**. That subnet gives it a public IPv4 address, so outbound traffic works, while the security group still allows nothing inbound.

With `--subnet subnet-… --vpc vpc-…`, the subnet needs outbound internet through a public IP or a NAT gateway. Setup downloads from GitHub, PyPI, astral.sh and awscli.amazonaws.com. SSM also needs to reach its endpoints, either over the internet or through the `ssm`, `ssmmessages` and `ec2messages` interface VPC endpoints. VPC endpoints alone are not enough, because setup needs the internet.

## Facilitator workflow

**Before the workshop**

1. Get the G and VT quota raised to at least 4 vCPUs × (participants + 1).
2. Deploy the budget (below).
3. Create one test stack and run a notebook on it (for example `prl-aws create test --cohort dryrun`), then destroy it.

**Create the cohort.** This creates `prl-oct26-01` through `prl-oct26-12` in parallel, resolving the AMI once for all of them:

```sh
infra/aws/bin/prl-aws cohort-up oct26 12 --region us-east-1
```

It prints the plan and warns when the quota looks too small. It asks for confirmation unless you pass `--yes`, skips stacks that already exist, and reports any stack that fails. `prl-aws status --cohort oct26` lists the cohort's instances.

**Giving participants access** is still open (OPEN_QUESTIONS C1). The options are:

- **(a) IAM Identity Center.** Each participant signs in with their own identity and gets a permission set that allows `ssm:StartSession` only on instances whose tags match. AWS's Session Manager sample policies show the `ssm:resourceTag/<key>` condition ([sample policies][ssm-policies]). The set also allows `ssm:GetParameter` on that participant's token parameter. Participants then run `prl-aws connect` themselves, with no long-lived keys. This needs an Identity Center setup.
- **(b) The facilitator runs port forwarding.** The facilitator's machine runs `prl-aws connect <NN> --port <88NN>` for each participant. The plugin listens only on the facilitator's own `localhost`, so this suits a demo or screen sharing more than independent work.
- **(c) SageMaker Studio presigned URLs (Option B).** The facilitator calls `CreatePresignedDomainUrl` for each participant's user profile, and participants need no AWS credentials. The URL must be opened within 5 minutes (300 seconds at most), and the session lasts up to 12 hours. The domain must use IAM authentication ([CreatePresignedDomainUrl][presigned]).

The working default in OPEN_QUESTIONS is (a) for teams that already use SSO and (c) otherwise. The decision belongs to the workshop owner.

**Tear down and check that nothing is left:**

```sh
infra/aws/bin/prl-aws cohort-down oct26 --region us-east-1
```

`cohort-down` finds every stack tagged `Project=practical-rl` and `Cohort=oct26`. It asks once, deletes them all, waits, and deletes their SSM parameters. Then it lists anything left from the cohort in that region:

- stacks, instances, volumes and security groups, found by tag;
- alarms, found by name prefix;
- SSM parameters under `/prl/prl-oct26-`;
- IAM roles and instance profiles whose names start with `prl-oct26-`.

It exits with an error if anything remains. Run it in every region you used. Check Billing a day later too, because costs appear with a delay.

**Budget.** Deploy the one-time account budget:

```sh
infra/aws/bin/prl-aws budget ops@example.com 300      # monthly limit in USD
```

This creates the stack `prl-account-budget`, a monthly COST budget for the **whole account**. It emails when actual spend passes 80% of the limit and when forecasted spend passes 100%. A budget only sends alerts; it does not stop anything.

`budget` deploys to **us-east-1** unless you pass `--region`. Whether `AWS::Budgets::Budget` *must* be deployed in us-east-1 is **unverified** (OPEN_QUESTIONS B21):

- The [resource reference][budget-ref] states no region requirement.
- The CloudFormation resource schemas bundled with cfn-lint 1.57.2 (schemas dated 2026-10-06) include the type in us-east-1, us-west-2 and eu-west-1, but not in eu-north-1. So the type is not limited to us-east-1, but it isn't available everywhere either.
- us-east-1 is the safe choice.

## Option B: SageMaker AI Studio JupyterLab space

This option suits facilitators who prefer a managed notebook, and it pairs with access option (c).

1. **Set up a domain.** Create a SageMaker AI domain (the quick setup is enough) and one user profile per participant.
2. **Turn on idle shutdown.** It requires the **SageMaker Distribution image version 2.0 or later** ([idle shutdown][sm-idle]). Set it for the domain or for each user profile ([set up idle shutdown][sm-idle-setup]). To enforce one timeout, set the three values equal, as in this AWS example:
   ```sh
   aws sagemaker update-domain --domain-id <domain-id> --default-user-settings '{
     "JupyterLabAppSettings": {"AppLifecycleManagement": {"IdleSettings": {
       "LifecycleManagement": "ENABLED", "IdleTimeoutInMinutes": 120,
       "MinIdleTimeoutInMinutes": 120, "MaxIdleTimeoutInMinutes": 120}}}}'
   ```
   Running apps must be restarted before the setting takes effect. For JupyterLab, "idle" means **no active kernel sessions and no active terminal sessions**. An open kernel keeps the space running, so participants should shut down their kernels.
3. **Create the space.** Each participant creates a JupyterLab space on `ml.g4dn.xlarge` with a SageMaker Distribution image of version 2.0 or later.
4. **Get the repository.** In the space, open a terminal and run `git clone https://github.com/project-delphi/practical-rl.git`.
5. **Install what the notebook needs.** Use `pip` in the space's terminal to install the packages that the notebook's first cell pins: `prl` and the few libraries it adds, such as `gymnasium` and `stable-baselines3` (PLAN.md §6). Keep the image's own torch, as on Colab. How that first cell behaves on SageMaker has not been defined yet (see [Unverified](#unverified)).

Classic SageMaker **notebook instances** are not a supported path here. Amazon Linux 2 notebook instances reached end of support on 2026-06-30, and since 2026-07-01 they can no longer be created or restarted. New notebook instances default to `notebook-al2023-v1` ([AL2 notebook instances][sm-al2]). The price table shows the notebook-instance price only for comparison.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `connect` says the Session Manager plugin is not installed | Install it ([plugin install][plugin-install]) and check `session-manager-plugin --version`. |
| `aws` fails with "exec format error" | The CLI binary is for the wrong architecture. Reinstall AWS CLI v2 ([install][cli-install]). |
| The stack fails with a vCPU limit error (`VcpuLimitExceeded`) | The G and VT quota is too small (see [Prerequisites](#prerequisites)). `destroy` the failed stack, request more vCPUs, and create it again. |
| `InsufficientInstanceCapacity`, or the instance type is unsupported in the Availability Zone | Try again later, try `--type g6.xlarge` or `--type g5.xlarge`, or pass `--subnet`/`--vpc` for a subnet in another Availability Zone. `start` can hit the same capacity error. |
| `create` says the stack is `ROLLBACK_COMPLETE` | `prl-aws` prints the failed events. Fix the cause, `destroy`, then `create` again. |
| `connect` warns that setup is still running, or that there is no token yet | Wait and run `prl-aws status <participant>`. If setup failed, open a shell with `aws ssm start-session --target <instance-id>` and read `sudo tail -n 100 /var/log/prl-setup.log`. After fixing the cause, re-run it with `sudo bash /var/lib/cloud/instance/user-data.txt` (cloud-init's copy of the user-data). It is idempotent and never touches an existing checkout. |
| The browser cannot reach `localhost:8888` | Keep the `connect` terminal open. On the instance, check `systemctl status prl-jupyter` and `journalctl -u prl-jupyter`. |
| The token is rejected | Copy the whole token from `connect`, or print it with the stack's `TokenCommand` output. The token stays the same across stops and starts. |
| The instance stopped by itself | The idle alarm did that. Run `start`, then `connect`. Your files are still there. |
| The alarm shows `INSUFFICIENT_DATA` | This is normal just after creation and while the instance is stopped. |
| The GPU is not used | Run `nvidia-smi` in a terminal, and `python -c "import torch; print(torch.cuda.is_available())"` in the project environment. See item 4 under [Unverified](#unverified). |

## Unverified

**Nothing here has been run end to end on AWS.** There were no AWS credentials, and the local `aws` CLI fails to run. The human end-to-end run is a release-checklist item (OPEN_QUESTIONS C2).

What *was* run on 2026-10-09:

- `infra/aws/lint.sh`: cfn-lint 1.57.2 on both templates; shellcheck 0.11.0 on `prl-aws`, `lint.sh` and the extracted user-data. All passed.
- `prl-aws`, run under macOS's bash 3.2 against a fake `aws` command, for every subcommand. This checks argument handling and the AWS CLI calls the script makes, not AWS's responses.
- `price_table.py`, once, for us-east-1.

Not verified:

1. **The user-data on the real DLAMI.** The DLAMI release notes do not list the AWS CLI, so user-data installs v2 when `aws` is missing. Also untested: the `ubuntu` user, `unzip` and the apt lock at first boot, and how long setup takes.
2. **The root device name.** `prl-aws` reads it from `describe-images`; the template defaults to `/dev/sda1`.
3. **The idle alarm.** Not checked: its timing, the stop action through CloudFormation, the service-linked role being created on demand, and the re-arm unit after a stop and start.
4. **CUDA in the project environment.** Whether the project's `uv.lock` gives a CUDA build of torch on Linux x86_64. If the uv configuration points Linux at a CPU-only PyTorch index (as for CI), the GPU goes unused on AWS. The `notebooks` group and the lock belong to the project, not to this directory.
5. **The SecureString key.** Whether the instance role and participants need any KMS permission. The parameter uses the AWS-managed `aws/ssm` key; we expect none.
6. **Parameter values on `--update`.** That `aws cloudformation deploy` keeps unspecified parameter values when it updates a stack.
7. **The default VPC.** How EC2 picks an Availability Zone for g4dn.xlarge when no subnet is given.
8. **The budget region** (B21, above). Also untested: how budget emails behave.
9. **Option B, end to end.** Also: the Python and torch versions in the current SageMaker Distribution image, and the name of SageMaker's own quota for JupyterLab apps on `ml.g4dn.xlarge`.
10. **The notebooks' setup cell outside Colab.** How it treats `PRL_PLATFORM=aws` on EC2 and how it behaves in a SageMaker space. The cell belongs to the notebook build, not to this directory.
11. **The cloud-init path** `/var/lib/cloud/instance/user-data.txt` used above for re-running setup.
12. **TRL's Triton kernel on a T4** (OPEN_QUESTIONS B22).

## Maintenance

- **Lint** after any change: `infra/aws/lint.sh`. It needs `uv`.
- **Prices:** run `python infra/aws/price_table.py`, then commit `prices.json` and this README. `--render-only` rebuilds the table from `prices.json`.
- **Pinned versions in the user-data** (`instance.yaml`): uv `0.12.24`, Python `3.13`, and JupyterLab `4.6.4`, which is used only when the project environment lacks it.
- **The AMI** is never pinned in the repository. `create` and `cohort-up` resolve the latest Base DLAMI at create time and print it. Pass `--ami` to reuse a known-good ID.

## Sources

Checked on 2026-10-09:

- [Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) release notes, 2026-10-08][dlami-notes]
- [EC2 instance quotas][ec2-quotas]; [Requesting a quota increase][quota-increase]
- [Accelerated computing instance specifications][ac-specs]
- [How EC2 stop and start works][ec2-stop-start]
- [CloudWatch alarm actions][alarm-actions-overview]; [Alarm actions for EC2][alarm-actions]; [Alarms and missing data][missing-data]
- [Session Manager plugin][plugin-install]; [Starting a port forwarding session][ssm-start]; [Session Manager sample IAM policies][ssm-policies]
- [AWS CLI v2 install][cli-install]
- [`AWS::EC2::Instance`][ec2-instance-ref]; [`AWS::Budgets::Budget`][budget-ref]
- [SageMaker idle shutdown][sm-idle]; [Set up idle shutdown][sm-idle-setup]; [AL2 notebook instances][sm-al2]; [CreatePresignedDomainUrl][presigned]
- [Amazon VPC pricing][vpc-pricing]
- [AWS Price List bulk API offer index](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/index.json)

[dlami-notes]: https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-ami-gpubaseoss-ul2404-2026-10-08.html
[ec2-quotas]: https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html
[quota-increase]: https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html
[ac-specs]: https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html
[ec2-stop-start]: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-stop-start-works.html
[alarm-actions-overview]: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/alarm-actions.html
[alarm-actions]: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/UsingAlarmActions.html
[missing-data]: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/alarms-and-missing-data.html
[plugin-install]: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html
[ssm-start]: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-sessions-start.html
[ssm-policies]: https://docs.aws.amazon.com/systems-manager/latest/userguide/getting-started-restrict-access-quickstart.html
[cli-install]: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
[ec2-instance-ref]: https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-ec2-instance.html
[budget-ref]: https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-budgets-budget.html
[sm-idle]: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-idle-shutdown.html
[sm-idle-setup]: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-idle-shutdown-setup.html
[sm-al2]: https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-al2.html
[presigned]: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreatePresignedDomainUrl.html
[vpc-pricing]: https://aws.amazon.com/vpc/pricing/
