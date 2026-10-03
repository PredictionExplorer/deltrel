# Next ESLint directory adapter

This private package replaces only `@next/eslint-plugin-next@16.3.8`'s
`fast-glob` dependency. That plugin calls exactly
`globSync(rootDir.replace(/\\/g, '/'), { onlyDirectories: true })` in
`dist/utils/get-root-dirs.js`. Its default root remains `context.cwd`; arrays
are still mapped and flattened by the unchanged Next plugin.

The original chain includes `braces@3.0.3`, affected by
[GHSA-vfj7-8cjw-p6xm](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm).
As of 2026-10-03 the advisory lists no patched release. This adapter removes
that chain while retaining Next, its lint rules, and the npm audit gate.
It uses the already-used, registry-pinned `tinyglobby@0.2.17` with directory
expansion explicitly disabled; a direct package alias would change behavior.
The root's development dependency named `fast-glob` points to this private
package so npm can resolve its local path reproducibly; the override references
that direct dependency with `$fast-glob` and applies only to the pinned Next
plugin. The lockfile retains the adapter's honest package name and local path.

Supported inputs are a single nonempty POSIX-style relative or absolute
directory path, `*`, `?`, globstars, character classes and comma brace lists.
Options must be exactly `{ onlyDirectories: true }`. Other APIs, brace ranges,
extglobs, backslash escapes and patterns over 4096 characters or 32 nesting
levels are outside this lint-only adapter's scope and throw. The Next helper
performs its existing Windows separator conversion before calling it.

Literal paths preserve their spelling and follow directory symlinks. Glob
results omit trailing slashes; a trailing globstar excludes its starting
directory. Explicit dot paths match, while implicit dot traversal stays off.
Directory symlink roots retain their lexical names. Traversal stops when a
descendant symlink returns to a physical ancestor within that traversal;
explicitly selecting an alias as the starting root still works. Permission
and other unexpected filesystem errors are surfaced, even though tinyglobby's
underlying walker normally suppresses them.
Matches within one glob are sorted. Fast-glob documents its match order as
arbitrary; the Next rule checks the union of discovered page/app routes.
The enclosing `rootDir` array retains input order and repeated roots.

This is **not** a general replacement for fast-glob. Recheck the audited
callsite and this compatibility suite before changing the scoped override or
Next ESLint version. Once upstream releases a compatible audited fix, remove
the adapter and override together. Run `npm run test:eslint-glob` to exercise
discovery, complexity limits, and the actual Next internal-link rule.
