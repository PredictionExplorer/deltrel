/* @ts-self-types="./deltrel_wasm.d.ts" */

/**
 * Browser-side deterministic Gumbel top-k/sequential-halving scheduler.
 */
export class WasmGumbel {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmGumbelFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmgumbel_free(ptr, 0);
    }
    /**
     * Whether the exact simulation budget has been consumed.
     * @returns {boolean}
     */
    done() {
        const ret = wasm.wasmgumbel_done(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Creates a root scheduler over the supplied stable logit order.
     * @param {Float32Array} logits
     * @param {number} simulations
     * @param {number} max_considered
     * @param {number} c_visit
     * @param {number} c_scale
     * @param {bigint} seed
     */
    constructor(logits, simulations, max_considered, c_visit, c_scale, seed) {
        const ptr0 = passArrayF32ToWasm0(logits, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmgumbel_new(ptr0, len0, simulations, max_considered, c_visit, c_scale, seed);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        WasmGumbelFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Next forced root edge, or `None` when complete.
     * @param {Float32Array} completed_q
     * @param {Uint32Array} visits
     * @returns {number | undefined}
     */
    next(completed_q, visits) {
        const ptr0 = passArrayF32ToWasm0(completed_q, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(visits, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmgumbel_next(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] === Number.MAX_SAFE_INTEGER ? undefined : ret[0];
    }
    /**
     * Next edge from the current phase, without transferring statistics.
     * Returns `None` at a phase boundary or after the budget is exhausted.
     * When not `done`, call `next` with current statistics to begin the next phase.
     * @returns {number | undefined}
     */
    next_scheduled() {
        const ret = wasm.wasmgumbel_next_scheduled(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * Records one completed root-edge simulation.
     * @param {number} candidate
     */
    record(candidate) {
        const ret = wasm.wasmgumbel_record(this.__wbg_ptr, candidate);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Selected root edge after search.
     * @param {Float32Array} completed_q
     * @param {Uint32Array} visits
     * @returns {number}
     */
    selected(completed_q, visits) {
        const ptr0 = passArrayF32ToWasm0(completed_q, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(visits, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmgumbel_selected(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
}
if (Symbol.dispose) WasmGumbel.prototype[Symbol.dispose] = WasmGumbel.prototype.free;

/**
 * Optional ordered first-visit batching and between-move subtree reuse.
 */
export class WasmSearchSession {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmSearchSessionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmsearchsession_free(ptr, 0);
    }
    /**
     * Stable legal root action order.
     * @returns {Uint16Array}
     */
    actions() {
        const ret = wasm.wasmsearchsession_actions(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU16FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 2, 2);
        return v1;
    }
    /**
     * Freeze one complete result for subsequent flat getters.
     */
    complete() {
        const ret = wasm.wasmsearchsession_complete(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Whether the new requested simulation budget is complete.
     * @returns {boolean}
     */
    done() {
        const ret = wasm.wasmsearchsession_done(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Work inherited before the new root prediction.
     * @returns {Uint32Array}
     */
    inherited_visits() {
        const ret = wasm.wasmsearchsession_inherited_visits(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Supply root predictions, refreshing retained statistics when reused.
     * @param {bigint} token
     * @param {number} value
     * @param {Float32Array} logits
     */
    initialize_root(token, value, logits) {
        const ptr0 = passArrayF32ToWasm0(logits, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsearchsession_initialize_root(this.__wbg_ptr, token, value, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Construct a fresh search; every supplied simulation is new work.
     * @param {WasmState} state
     * @param {number} simulations
     * @param {number} max_considered
     * @param {number} c_visit
     * @param {number} c_scale
     * @param {bigint} seed
     * @param {number} first_visit_batch_size
     */
    constructor(state, simulations, max_considered, c_visit, c_scale, seed, first_visit_batch_size) {
        _assertClass(state, WasmState);
        const ret = wasm.wasmsearchsession_new(state.__wbg_ptr, simulations, max_considered, c_visit, c_scale, seed, first_visit_batch_size);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        WasmSearchSessionFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Obtain the next owned batch; submit it before asking for another.
     * @returns {number}
     */
    next_requests() {
        const ret = wasm.wasmsearchsession_next_requests(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Legal placements for one pending row.
     * @param {number} index
     * @returns {Uint16Array}
     */
    pending_actions(index) {
        const ret = wasm.wasmsearchsession_pending_actions(this.__wbg_ptr, index);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU16FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 2, 2);
        return v1;
    }
    /**
     * Copy one pending state; the JavaScript caller owns the returned state.
     * @param {number} index
     * @returns {WasmState}
     */
    pending_state(index) {
        const ret = wasm.wasmsearchsession_pending_state(this.__wbg_ptr, index);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmState.__wrap(ret[0]);
    }
    /**
     * Opaque tokens in pending row order.
     * @returns {BigUint64Array}
     */
    pending_tokens() {
        const ret = wasm.wasmsearchsession_pending_tokens(this.__wbg_ptr);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Policy target with the new-budget sigma scale.
     * @returns {Float32Array}
     */
    policy_target() {
        const ret = wasm.wasmsearchsession_policy_target(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Q estimates from inherited and new evidence.
     * @returns {Float32Array}
     */
    q_values() {
        const ret = wasm.wasmsearchsession_q_values(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Begin another root, reusing only eligible exact proper descendants.
     * @param {WasmState} state
     * @param {number} simulations
     * @param {number} max_considered
     * @param {number} c_visit
     * @param {number} c_scale
     * @param {bigint} seed
     * @param {number} first_visit_batch_size
     * @param {boolean} allow_reuse
     * @param {number} max_nodes
     */
    restart(state, simulations, max_considered, c_visit, c_scale, seed, first_visit_batch_size, allow_reuse, max_nodes) {
        _assertClass(state, WasmState);
        const ret = wasm.wasmsearchsession_restart(this.__wbg_ptr, state.__wbg_ptr, simulations, max_considered, c_visit, c_scale, seed, first_visit_batch_size, allow_reuse, max_nodes);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Retained node count at restart.
     * @returns {number}
     */
    reused_nodes() {
        const ret = wasm.wasmsearchsession_reused_nodes(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Retained root outgoing visits at restart.
     * @returns {number}
     */
    reused_visits() {
        const ret = wasm.wasmsearchsession_reused_visits(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Root inference is required for both a fresh and a reused root.
     * @returns {Uint16Array}
     */
    root_actions() {
        const ret = wasm.wasmsearchsession_root_actions(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU16FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 2, 2);
        return v1;
    }
    /**
     * Token for the current root prediction.
     * @returns {bigint}
     */
    root_token() {
        const ret = wasm.wasmsearchsession_root_token(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return BigInt.asUintN(64, ret[0]);
    }
    /**
     * Diagnostic mean over inherited and new root edge evidence.
     * @returns {number | undefined}
     */
    root_value() {
        const ret = wasm.wasmsearchsession_root_value(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] === Number.MAX_SAFE_INTEGER ? undefined : ret[0];
    }
    /**
     * Selected placement, absent only for a terminal root.
     * @returns {number | undefined}
     */
    selected_action() {
        const ret = wasm.wasmsearchsession_selected_action(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] === 0xFFFFFF ? undefined : ret[0];
    }
    /**
     * Selected keep-continuation value in root-player perspective.
     * @returns {number | undefined}
     */
    selected_action_value() {
        const ret = wasm.wasmsearchsession_selected_action_value(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] === Number.MAX_SAFE_INTEGER ? undefined : ret[0];
    }
    /**
     * New simulations completed during this root search.
     * @returns {number}
     */
    simulations() {
        const ret = wasm.wasmsearchsession_simulations(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Validate the whole flat response batch before any ordered backup.
     * @param {BigUint64Array} tokens
     * @param {Float32Array} values
     * @param {Uint32Array} offsets
     * @param {Float32Array} logits
     */
    submit(tokens, values, offsets, logits) {
        const ptr0 = passArray64ToWasm0(tokens, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(values, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArray32ToWasm0(offsets, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passArrayF32ToWasm0(logits, wasm.__wbindgen_malloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsearchsession_submit(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Combined inherited and new root edge visits.
     * @returns {Uint32Array}
     */
    total_visits() {
        const ret = wasm.wasmsearchsession_total_visits(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Current tree size, for bounding retained browser memory.
     * @returns {number}
     */
    unique_nodes() {
        const ret = wasm.wasmsearchsession_unique_nodes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * New root edge visits, whose sum is the new requested budget.
     * @returns {Uint32Array}
     */
    visits() {
        const ret = wasm.wasmsearchsession_visits(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
}
if (Symbol.dispose) WasmSearchSession.prototype[Symbol.dispose] = WasmSearchSession.prototype.free;

/**
 * Ask/tell atomic-action search tree for browser inference.
 */
export class WasmSearchTree {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmSearchTreeFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmsearchtree_free(ptr, 0);
    }
    /**
     * Root node ids in stable order.
     * @returns {Uint16Array}
     */
    actions() {
        const ret = wasm.wasmsearchtree_actions(this.__wbg_ptr);
        var v1 = getArrayU16FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 2, 2);
        return v1;
    }
    /**
     * Completed root Q values in stable order.
     * @returns {Float32Array}
     */
    completed_q() {
        const ret = wasm.wasmsearchtree_completed_q(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Supplies one pending leaf evaluation.
     * @param {bigint} token
     * @param {number} value
     * @param {Float32Array} policy_logits
     */
    finish(token, value, policy_logits) {
        const ptr0 = passArrayF32ToWasm0(policy_logits, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsearchtree_finish(this.__wbg_ptr, token, value, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Supplies initial root inference.
     * @param {bigint} token
     * @param {number} value
     * @param {Float32Array} policy_logits
     */
    initialize_root(token, value, policy_logits) {
        const ptr0 = passArrayF32ToWasm0(policy_logits, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmsearchtree_initialize_root(this.__wbg_ptr, token, value, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Creates a search tree from a state snapshot.
     * @param {WasmState} state
     * @param {number} c_visit
     * @param {number} c_scale
     */
    constructor(state, c_visit, c_scale) {
        _assertClass(state, WasmState);
        const ret = wasm.wasmsearchtree_new(state.__wbg_ptr, c_visit, c_scale);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        WasmSearchTreeFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Stable pending legal actions required by `finish`.
     * @returns {Uint16Array}
     */
    pending_actions() {
        const ret = wasm.wasmsearchtree_pending_actions(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU16FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 2, 2);
        return v1;
    }
    /**
     * Pending leaf state.
     * @returns {WasmState}
     */
    pending_state() {
        const ret = wasm.wasmsearchtree_pending_state(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmState.__wrap(ret[0]);
    }
    /**
     * Token that must accompany the pending leaf response.
     * @returns {bigint}
     */
    pending_token() {
        const ret = wasm.wasmsearchtree_pending_token(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return BigInt.asUintN(64, ret[0]);
    }
    /**
     * Whether root values are reported as the opener's optimal-swap payoff.
     * @returns {boolean}
     */
    pie_root_transform() {
        const ret = wasm.wasmsearchtree_pie_root_transform(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Completed-Q policy target in stable order.
     * @returns {Float32Array}
     */
    policy_target() {
        const ret = wasm.wasmsearchtree_policy_target(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Stable root legal node ids required by `initialize_root`.
     * @returns {Uint16Array}
     */
    root_actions() {
        const ret = wasm.wasmsearchtree_root_actions(this.__wbg_ptr);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU16FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 2, 2);
        return v1;
    }
    /**
     * Token that must accompany the root evaluation response.
     * @returns {bigint}
     */
    root_token() {
        const ret = wasm.wasmsearchtree_root_token(this.__wbg_ptr);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return BigInt.asUintN(64, ret[0]);
    }
    /**
     * Visit-weighted root value for the player to move, once any
     * simulation has completed. This includes exploration; use the
     * selected edge's `completed_q` for the responder's pie-swap decision.
     * @returns {number | undefined}
     */
    root_value() {
        const ret = wasm.wasmsearchtree_root_value(this.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * Starts one simulation, forcing the supplied root action.
     *
     * Returns `true` when leaf inference is required and `false` when an
     * exact terminal value was backed up immediately.
     * @param {number} root_action
     * @returns {boolean}
     */
    start(root_action) {
        const ret = wasm.wasmsearchtree_start(this.__wbg_ptr, root_action);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] !== 0;
    }
    /**
     * Root edge visit counts in stable order.
     * @returns {Uint32Array}
     */
    visits() {
        const ret = wasm.wasmsearchtree_visits(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
}
if (Symbol.dispose) WasmSearchTree.prototype[Symbol.dispose] = WasmSearchTree.prototype.free;

/**
 * Browser-owned Deltrel state for any rule variant.
 */
export class WasmState {
    static __wrap(ptr) {
        const obj = Object.create(WasmState.prototype);
        obj.__wbg_ptr = ptr;
        WasmStateFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmStateFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmstate_free(ptr, 0);
    }
    /**
     * Applies one dense node id.
     * @param {number} node
     */
    apply(node) {
        const ret = wasm.wasmstate_apply(this.__wbg_ptr, node);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Placements of the unfinished current turn.
     * @returns {BigUint64Array}
     */
    current_turn_bits() {
        const ret = wasm.wasmstate_current_turn_bits(this.__wbg_ptr);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Placements of the turn in progress: handicap during the opening.
     * @returns {number}
     */
    get current_turn_total() {
        const ret = wasm.wasmstate_current_turn_total(this.__wbg_ptr);
        return ret;
    }
    /**
     * Consecutive opening placements by player 0.
     * @returns {number}
     */
    get handicap() {
        const ret = wasm.wasmstate_handicap(this.__wbg_ptr);
        return ret;
    }
    /**
     * Stones placed during the opening phase.
     * @returns {BigUint64Array}
     */
    handicap_bits() {
        const ret = wasm.wasmstate_handicap_bits(this.__wbg_ptr);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Stable deterministic state hash.
     * @returns {bigint}
     */
    hash64() {
        const ret = wasm.wasmstate_hash64(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * Legal node ids in ascending order.
     * @returns {Uint16Array}
     */
    legal_actions() {
        const ret = wasm.wasmstate_legal_actions(this.__wbg_ptr);
        var v1 = getArrayU16FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 2, 2);
        return v1;
    }
    /**
     * Legal placement bitboard words.
     * @returns {BigUint64Array}
     */
    legal_bits() {
        const ret = wasm.wasmstate_legal_bits(this.__wbg_ptr);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Largest supported handicap.
     * @returns {number}
     */
    static max_handicap() {
        const ret = wasm.wasmstate_max_handicap();
        return ret;
    }
    /**
     * Turn protocol name: `"classic"` or `"double"`.
     * @returns {string}
     */
    get mode() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmstate_mode(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Placements remaining in this turn.
     * @returns {number}
     */
    get moves_left() {
        const ret = wasm.wasmstate_moves_left(this.__wbg_ptr);
        return ret;
    }
    /**
     * Creates an empty state. `mode` is `"classic"` or `"double"`,
     * `handicap` is `1..=9`, and `pie` enables the swap after the opening.
     * @param {number} rings
     * @param {string} mode
     * @param {number} handicap
     * @param {boolean} pie
     */
    constructor(rings, mode, handicap, pie) {
        const ptr0 = passStringToWasm0(mode, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmstate_new(rings, ptr0, len0, handicap, pie);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        WasmStateFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Player-one fixed bitboard words.
     * @returns {BigUint64Array}
     */
    one_bits() {
        const ret = wasm.wasmstate_one_bits(this.__wbg_ptr);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Whether the opening turn is active.
     * @returns {boolean}
     */
    get opening() {
        const ret = wasm.wasmstate_opening(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * The completed turn before that (the mover's).
     * @returns {BigUint64Array}
     */
    own_previous_turn_bits() {
        const ret = wasm.wasmstate_own_previous_turn_bits(this.__wbg_ptr);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Whether the pie rule is in effect.
     * @returns {boolean}
     */
    get pie() {
        const ret = wasm.wasmstate_pie(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Empty board of a pie game.
     * @returns {boolean}
     */
    get pie_pending() {
        const ret = wasm.wasmstate_pie_pending(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * The most recently completed turn (the opponent's).
     * @returns {BigUint64Array}
     */
    previous_turn_bits() {
        const ret = wasm.wasmstate_previous_turn_bits(this.__wbg_ptr);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Rules-contract hash expected by replay data and services.
     * @returns {bigint}
     */
    static rules_hash() {
        const ret = wasm.wasmstate_rules_hash();
        return BigInt.asUintN(64, ret);
    }
    /**
     * Tagged finalized rules hash.
     * @returns {string}
     */
    static rules_hash_tag() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmstate_rules_hash_tag();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Finalized rules schema.
     * @returns {string}
     */
    static rules_schema() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.wasmstate_rules_schema();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Fourteen score integers: six per player, contested, leader.
     * @returns {Int32Array}
     */
    score_components() {
        const ret = wasm.wasmstate_score_components(this.__wbg_ptr);
        var v1 = getArrayI32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Creates an empty standard Double Deltrel state.
     * @param {number} rings
     * @returns {WasmState}
     */
    static standard(rings) {
        const ret = wasm.wasmstate_standard(rings);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmState.__wrap(ret[0]);
    }
    /**
     * Applies the pie swap; legal only right after the opening turn.
     */
    swap() {
        const ret = wasm.wasmstate_swap(this.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Whether player 1 may swap right now.
     * @returns {boolean}
     */
    get swap_available() {
        const ret = wasm.wasmstate_swap_available(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether the swap was taken earlier in this game.
     * @returns {boolean}
     */
    get swapped() {
        const ret = wasm.wasmstate_swapped(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Terminal marker.
     * @returns {boolean}
     */
    get terminal() {
        const ret = wasm.wasmstate_terminal(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Current player's numeric index.
     * @returns {number}
     */
    get to_move() {
        const ret = wasm.wasmstate_to_move(this.__wbg_ptr);
        return ret;
    }
    /**
     * D5-transformed copy.
     * @param {number} symmetry
     * @returns {WasmState}
     */
    transformed(symmetry) {
        const ret = wasm.wasmstate_transformed(this.__wbg_ptr, symmetry);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmState.__wrap(ret[0]);
    }
    /**
     * Player-zero fixed bitboard words.
     * @returns {BigUint64Array}
     */
    zero_bits() {
        const ret = wasm.wasmstate_zero_bits(this.__wbg_ptr);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
}
if (Symbol.dispose) WasmState.prototype[Symbol.dispose] = WasmState.prototype.free;

/**
 * Search behavior fingerprint, independent of game and model schemas.
 * @returns {string}
 */
export function search_algorithm_id() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.search_algorithm_id();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Version of the optional batched and persistent search execution API.
 * @returns {number}
 */
export function search_execution_version() {
    const ret = wasm.search_execution_version();
    return ret >>> 0;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_344f42d3211c4765: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./deltrel_wasm_bg.js": import0,
    };
}

const WasmGumbelFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmgumbel_free(ptr, 1));
const WasmSearchSessionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmsearchsession_free(ptr, 1));
const WasmSearchTreeFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmsearchtree_free(ptr, 1));
const WasmStateFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmstate_free(ptr, 1));

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayI32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU16FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint16ArrayMemory0().subarray(ptr / 2, ptr / 2 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getBigUint64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

let cachedBigUint64ArrayMemory0 = null;
function getBigUint64ArrayMemory0() {
    if (cachedBigUint64ArrayMemory0 === null || cachedBigUint64ArrayMemory0.byteLength === 0) {
        cachedBigUint64ArrayMemory0 = new BigUint64Array(wasm.memory.buffer);
    }
    return cachedBigUint64ArrayMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedInt32ArrayMemory0 = null;
function getInt32ArrayMemory0() {
    if (cachedInt32ArrayMemory0 === null || cachedInt32ArrayMemory0.byteLength === 0) {
        cachedInt32ArrayMemory0 = new Int32Array(wasm.memory.buffer);
    }
    return cachedInt32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint16ArrayMemory0 = null;
function getUint16ArrayMemory0() {
    if (cachedUint16ArrayMemory0 === null || cachedUint16ArrayMemory0.byteLength === 0) {
        cachedUint16ArrayMemory0 = new Uint16Array(wasm.memory.buffer);
    }
    return cachedUint16ArrayMemory0;
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getBigUint64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedBigUint64ArrayMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedInt32ArrayMemory0 = null;
    cachedUint16ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('deltrel_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
