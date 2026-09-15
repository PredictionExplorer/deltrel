# Reserve one Lambda 8× B200 instance

The standalone [Python watcher](../scripts/lambda_capacity_watch.py) waits for
`gpu_8x_b200_sxm6` capacity in **any region**, then launches one instance and exits.
It uses Python 3.11+ and the standard library on macOS or Linux. No installation
of training dependencies is required.

On Lambda, this is an actual paid instance launch, not a free reservation queue.
Billing begins after the instance passes health checks and continues while it
is idle, until you terminate it. Stopping this watcher does not terminate a VM.

## Quick start

Create a Lambda API key and register your SSH public key in the intended Lambda
workspace. Use [Lambda's console instructions](https://docs.lambda.ai/public-cloud/console/).
The SSH key argument below is the key's **name in Lambda**, not its local file path.

From the repository root, list the available regions, current whole-instance
price, and your registered SSH key names:

```bash
python3 training/scripts/lambda_capacity_watch.py --list-options
```

When `LAMBDA_API_KEY` is not set, an interactive terminal prompts for the key
without displaying it. The script does not write the key to disk or logs.
For unattended operation, supply `LAMBDA_API_KEY` through your process manager's
protected environment or secret facility. Do not put the key in command-line
arguments, source control, or a shared shell-history entry.

Start the watcher, replacing `YOUR_LAMBDA_SSH_KEY_NAME`:

```bash
python3 training/scripts/lambda_capacity_watch.py \
  --ssh-key YOUR_LAMBDA_SSH_KEY_NAME \
  --max-hourly-usd 60
```

The default ceiling is **$60 per hour for the entire eight-GPU instance**.
The watcher checks Lambda's reported whole-instance price immediately before
attempting a launch. This is a quoted-price check, not a total-spend limit or a
price-lock feature of Lambda's launch API; taxes and actual billing remain
subject to Lambda's terms. It never substitutes a different instance type.

Successful output includes the instance ID and region. The watcher exits and
leaves that instance running for you to use. View its IP and boot status in the
Lambda console. Training installation and data migration are separate steps.

## Polling and error handling

- One capacity check every **300–330 seconds**, approximately 11–12 per hour.
  Startup, pre-launch duplicate checks, and uncertain-launch reconciliation use
  a small number of additional requests.
- Requests are sequential, with at least **two seconds between request starts**.
  The client also enforces at least 15 seconds between explicit launch requests;
  normal watcher launch attempts are separated by the polling interval.
- Temporary read failures and HTTP 429 responses use exponential backoff,
  growing to roughly an hour. A longer `Retry-After` value is honored. The
  cooldown and backoff survive process restarts.
- A documented insufficient-capacity rejection resumes normal polling. It
  does not rapidly try every region or hammer the launch endpoint.
- Authentication, account, quota, and malformed-response problems fail clearly
  when they can be classified safely.
- There are no automatic HTTP-level retries of the launch POST.

Lambda documents general API limits of one request per second and one launch
request per 12 seconds. These defaults are comfortably below those limits.
They cannot guarantee an account will never be restricted; provider policies
and other processes using the same account also apply.

## Duplicate protection and restarting

The default durable journal is:

```text
~/.local/state/edgeconnect/lambda-b200-watch.json
```

Its adjacent lock file prevents two local copies using this journal from
running simultaneously. State files use private permissions and atomic,
flushed writes. **Launch intent reaches disk before the POST is sent.** After
success, restarting with the same journal returns the recorded reservation
without launching another VM, even if that VM was subsequently terminated.

Lambda does not document an idempotency guarantee for launch. If the connection
fails, the response is ambiguous, or the process is interrupted after saving
intent, the watcher checks the instance list for the recorded name/type/region.
It sends **no more launch requests** while that intent remains unresolved.
An empty instance list is never treated as proof that the launch failed.

If the outcome remains unconfirmed, inspect the Lambda console before resetting
anything. A crash immediately before the POST can also leave this conservative
pending state. Keep the journal while determining whether an instance exists.
Do not put the watcher under a supervisor that deletes its state on restart.

Run **one watcher on one host**. Local locks cannot coordinate different hosts
or different state files, and Lambda instance names are not guaranteed unique.
An existing matching instance is recognized; incompatible or duplicate names
stop the watcher for inspection. For an intentionally separate reservation,
use both a new `--name` and a separate `--state-file` after reviewing the existing
instance and charges.

Ctrl+C stops the watcher; rerun the same command to resume. For a long wait, run
it in a persistent terminal such as `tmux` on an always-on machine. A sleeping
laptop cannot poll. Avoid automatic restart-on-success: successful exit means
the reservation is already recorded.

## Optional controls

Check once without sending a launch request:

```bash
python3 training/scripts/lambda_capacity_watch.py \
  --ssh-key YOUR_LAMBDA_SSH_KEY_NAME --dry-run
```

This preserves any existing journal and cooldown. A dry run with pending launch
intent performs only reconciliation; it does not discard or retry that intent.

Restrict regions if you later choose to:

```bash
python3 training/scripts/lambda_capacity_watch.py \
  --ssh-key YOUR_LAMBDA_SSH_KEY_NAME \
  --region us-south-2 --region us-west-1
```

Purchase settings must match an existing journal. Decide the region list,
price ceiling, SSH key, and name before starting; do not reuse a pending journal
for a different purchase. `--interval-seconds` can change across restarts; its
minimum is 120 seconds. The default is recommended.

The launch uses Lambda's default image and firewall configuration. It does not
attach a persistent filesystem. Lambda filesystems must be in the same region
and workspace as the new instance and must be attached **at launch**, so this
any-region reservation does not automatically inherit the current training
server's filesystem. Plan the subsequent data transfer or storage setup before
moving training.

## Validation and sources

Tests use fake API responses and loopback HTTP servers with fake credentials.
They cover capacity races, quote checks, region selection, malformed state,
duplicate protection, interruption and disk failures, rate limiting, redirects,
credential redaction, and read-only reconciliation. No live capacity purchase
is needed to run them:

```bash
cd training
.venv/bin/pytest tests/test_lambda_capacity_watch.py tests/test_lambda_capacity_http.py
```

API contract checked against [Lambda's API reference](https://docs-api.lambda.ai/api/cloud),
[filesystem rules](https://docs.lambda.ai/public-cloud/filesystems/), and
[billing documentation](https://docs.lambda.ai/public-cloud/billing/).
