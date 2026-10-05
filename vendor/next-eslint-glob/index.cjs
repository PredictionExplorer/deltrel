'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { globSync: directories, isDynamicPattern } = require('tinyglobby');

// This deliberately implements only Next's audited rootDir callsite, not the
// general fast-glob API. Reject API or pattern scope changes explicitly.
function validate(pattern, options) {
  if (typeof pattern !== 'string' || pattern.length === 0 || pattern.length > 4096) {
    throw new TypeError('Next ESLint rootDir must be a nonempty string of at most 4096 characters');
  }
  const keys = options && typeof options === 'object' ? Reflect.ownKeys(options) : [];
  if (keys.length !== 1 || keys[0] !== 'onlyDirectories' || options.onlyDirectories !== true) {
    throw new TypeError('Next ESLint glob adapter supports only { onlyDirectories: true }');
  }
  const nesting = [];
  const braces = [];
  for (let i = 0; i < pattern.length; i++) {
    const char = pattern[i];
    if (char === '\\' || char === '\0') throw new TypeError('Next ESLint rootDir must use forward slashes');
    if (char === '{' || char === '(' || char === '[') {
      nesting.push(char);
      if (nesting.length > 32) throw new RangeError('Next ESLint rootDir nesting exceeds 32');
    } else if ({ '}': '{', ')': '(', ']': '[' }[char] === nesting.at(-1)) {
      nesting.pop();
    }
    if (char === '{') braces.push(i);
    if (char === '}' && braces.length) {
      const body = pattern.slice(braces.pop() + 1, i);
      if (body.includes('..') && !body.includes(',')) {
        throw new TypeError('Next ESLint rootDir supports brace lists, not brace ranges');
      }
    }
    if ('?*+@!'.includes(char) && pattern[i + 1] === '(') {
      throw new TypeError('Next ESLint rootDir extglobs are outside this adapter API');
    }
  }
}

function directoryReader() {
  const resolved = new Map();
  let traversalRoot;
  let failure;
  const realpath = value => {
    if (!resolved.has(value)) resolved.set(value, fs.realpathSync(value));
    return resolved.get(value);
  };
  const read = (directory, options) => {
    // Keep a symlink's lexical name without following cycles indefinitely.
    const absolute = path.resolve(directory), physical = realpath(absolute);
    traversalRoot ??= absolute;
    const prefix = traversalRoot.endsWith(path.sep) ? traversalRoot : traversalRoot + path.sep;
    for (let parent = path.dirname(absolute); absolute !== traversalRoot &&
      (parent === traversalRoot || parent.startsWith(prefix));) {
      if (realpath(parent) === physical) return [];
      const next = path.dirname(parent);
      if (next === parent) break;
      parent = next;
    }
    return fs.readdirSync(directory, options).map(entry => {
      if (!entry.isSymbolicLink()) return entry;
      try {
        if (!fs.statSync(path.join(directory, entry.name)).isDirectory()) return entry;
      } catch (error) {
        if (['ENOENT', 'ENOTDIR'].includes(error.code)) return entry;
        throw error;
      }
      // fdir otherwise traverses directory symlinks but omits their root entry.
      return Object.assign(Object.create(entry), {
        isDirectory: () => true,
        isSymbolicLink: () => false,
      });
    });
  };
  return {
    readdirSync(directory, options) {
      try { return read(directory, options); }
      catch (error) {
        // fdir suppresses filesystem errors by default. Missing entries are
        // expected; permission/IO failures must not silently erase lint roots.
        if (!['ENOENT', 'ENOTDIR'].includes(error.code)) failure ??= error;
        throw error;
      }
    },
    check() { if (failure) throw failure; },
  };
}

function globSync(pattern, options) {
  validate(pattern, options);
  if (!isDynamicPattern(pattern)) {
    // Preserve literal spelling (./, trailing slash, absolute path) and follow
    // directory symlinks, just like fast-glob's static directory lookup.
    try { return fs.statSync(pattern).isDirectory() ? [pattern] : []; }
    catch (error) {
      if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return [];
      throw error;
    }
  }
  // fast-glob excludes the starting directory of a final globstar. tinyglobby
  // otherwise includes it, so require one descendant entry explicitly.
  const normalized = pattern.replace(/\/+$/, '').replace(/(^|\/)\*\*$/, '$1**/*');
  const reader = directoryReader();
  const matches = directories(normalized, {
    cwd: process.cwd(),
    absolute: path.isAbsolute(pattern),
    onlyDirectories: true,
    expandDirectories: false,
    followSymbolicLinks: true,
    fs: { readdirSync: reader.readdirSync },
  });
  reader.check();
  return matches.map(value => value === '/' ? value : value.replace(/\/+$/, '')).sort();
}

module.exports = Object.freeze({ globSync });
