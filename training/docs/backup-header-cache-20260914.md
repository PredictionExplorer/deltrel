# Operation-scoped disaster-recovery header reuse

The September 14 deployment encountered a backup that spent over 20 minutes
processing historical snapshot catalogs while training continued. Read-only
inspection found 930 catalog documents totaling 2.63 GB. The process used about
one CPU core and 27 GB RSS after most payload I/O had completed. It published
its new catalog at 00:33:19 UTC, the global latest pointer at 00:36:48, and the
commit marker at 00:42:45, then exited successfully.

`create_snapshot` resolves the previous per-run/global latest pointers, checks
historical timestamps, and revalidates the updated pointers. For a single run
namespace, these steps scan the historical directory seven times. Previously,
every scan fully parsed, canonicalized, hashed, and expanded every historical
catalog again. Some nested calls also retained separate expanded copies.

The change memoizes validated headers only for one `create_snapshot`,
`verify_snapshot`, or `garbage_collect` invocation. Nested operations share the
same context; successful completion and exceptions both discard it. Every
lookup checks device, inode, size, modification time, change time, and backup
namespace **and hashes the current document bytes with SHA256**. Reuse requires
the current digest and length to equal the validated document. Replaced or
changed documents receive full validation again; deleted documents lose their
cached proof. A fresh byte check after initial parsing also detects a concurrent
content change before exposing that proof.
Cached documents and their nested metadata are read-only, and catalog entries
are frozen.

An initial metadata-only implementation was withdrawn before support-service
activation. The target-host test demonstrated actual same-size byte changes
with restored modification time and colliding change-time timestamps. Current
byte verification removes that timestamp assumption; deterministic tests also
force identical stat signatures while corrupting the bytes.

Directory contents, latest pointers, and commit markers are reread normally.
New, uncommitted, committed, missing, and corrupt documents retain their prior
selection semantics. Actual object/catalog integrity verification is unchanged
and is never replaced by a cached header. No durable format or cross-operation
trust cache is introduced.

## Local component measurement

The [benchmark record](backup-header-cache-benchmark-20260914.json) records three
repeats on local immutable synthetic catalogs: 64 documents, 1,500 entries each,
and eight header scans per operation. Both cases used the same current header
validator; the comparison enabled or disabled operation-level reuse. Setup was
excluded. All passes produced the same validated header summary digest.

| Measure | Without reuse | With reuse |
|---|---:|---:|
| Median CPU time | 3.295 s | 0.556 s |
| Median elapsed time | 3.419 s | 0.589 s |
| Full envelope validations | 512 | 64 |
| Catalog bytes parsed | 109.35 MB | 13.67 MB |
| Current-byte SHA256 checks | 512 | 512 |
| Current document bytes rehashed | 109.35 MB | 109.35 MB |

This is a **5.93× reduction in CPU time for the measured header-selection
component**. It is not a measurement of complete backup duration, NFS behavior,
model training throughput, or Elo. The first scan still fully validates all
historical catalogs; all subsequent scans still hash the current bytes. Payload
copying/verification remains necessary. Benchmark revision 2 supersedes the
withdrawn metadata-only measurement.

Tests cover immutable return values, nested contexts, exception cleanup,
replacement, corruption with restored modification time, concurrent change,
deletion, symlinks, namespace changes, newly committed snapshots, malformed and
missing latest pointers, and complete snapshot/restore integrity behavior.

Deployment can update the inactive disaster-backup service's reader without
restarting training. Its support-tool source commit should be recorded
separately from the running training release and profile authority.
