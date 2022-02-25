//! # PoS validity predicate tests
//!
//! The testing strategy is heavily relying on
//! [proptest state machine testing](https://github.com/AltSysrq/proptest/pull/257)
//! together with
//! [higher-order strategies](https://altsysrq.github.io/proptest-book/proptest/tutorial/higher-order.html).
//!
//! The system is being tested with arbitrary valid PoS parameters, arbitrary
//! valid initial state and:
//!
//! 1. One or more arbitrary valid transition. The happy path, these are very
//!    simple. We just test that they get accepted.
//! 1. One or more arbitrary valid transition, modified to make at least one of
//!    them invalid. Must be rejected.
//! 1. Arbitrary invalid storage modification. Must be rejected.
//!
//! The order of the arbitrary valid transitions is important as their storage
//! changes are accumulative, the same way they would be when applied in a
//! transaction. So for example we can initialize a validator account and bond
//! some tokens to it in the same transaction, but we cannot bond tokens to a
//! validator account before it's initialized.
//!
//! ## Pos Parameters
//!
//! Arbitrary valid PoS parameters are provided from its module via
//! [`anoma_vm_env::proof_of_stake::parameters::testing::arb_pos_params`].
//!
//! ## Valid transitions
//!
//! The list below includes state requirements and the storage key changes
//! expected for a valid transition. Valid transitions can be composed into an
//! ordered sequence. The composed transitions must still be a valid transition,
//! provided all the state requirements are valid (a transition may depend on
//! the modifications of its predecessor transition).
//!
//! The PoS storage modifications are modelled using
//! [`testing::PosStorageChange`].
//!
//! - Bond: Requires a validator account in the state (the `#{validator}`
//!   segments in the keys below). Some of the storage change are optional,
//!   which depends on whether the bond increases voting power of the validator.
//!     - `#{PoS}/bond/#{owner}/#{validator}`
//!     - `#{PoS}/total_voting_power` (optional)
//!     - `#{PoS}/validator_set` (optional)
//!     - `#{PoS}/validator/#{validator}/total_deltas`
//!     - `#{PoS}/validator/#{validator/voting_power` (optional)
//!     - `#{staking_token}/balance/#{PoS}`
//!
//!
//! - Unbond: Requires a bond in the state (the `#{owner}` and `#{validator}`
//!   segments in the keys below must be the owner and a validator of an
//!   existing bond). The bond's total amount must be greater or equal to the
//!   amount that' being unbonded. Some of the storage changes are optional,
//!   which depends on whether the unbonding decreases voting power of the
//!   validator.
//!     - `#{PoS}/bond/#{owner}/#{validator}`
//!     - `#{PoS}/total_voting_power` (optional)
//!     - `#{PoS}/unbond/#{owner}/#{validator}`
//!     - `#{PoS}/validator_set` (optional)
//!     - `#{PoS}/validator/#{validator}/total_deltas`
//!     - `#{PoS}/validator/#{validator}/voting_power` (optional)
//!
//! - Withdraw: Requires a withdrawable unbond in the state (the `#{owner}` and
//!   `#{validator}` segments in the keys below must be the owner and a
//!   validator of an existing withdrawable unbond).
//!     - `#{PoS}/unbond/#{owner}/#{validator}`
//!     - `#{staking_token}/balance/#{owner}`
//!     - `#{staking_token}/balance/#{PoS}`
//!
//! - Init validator: No state requirements.
//!     - `#{PoS}/address_raw_hash/{raw_hash}` (the raw_hash is the validator's
//!       address in Tendermint)
//!     - `#{PoS}/validator_set`
//!     - `#{PoS}/validator/#{validator}/consensus_key`
//!     - `#{PoS}/validator/#{validator}/staking_reward_address`
//!     - `#{PoS}/validator/#{validator}/state`
//!     - `#{PoS}/validator/#{validator}/total_deltas`
//!     - `#{PoS}/validator/#{validator}/voting_power`
//!
//!
//! ## Invalidating transitions
//!
//! To look for vulnerabilities in the VP, we can make arbitrary
//! [`crate::storage::Change`]s to the valid transitions that should
//! invalidate it and check that the VP does no longer accept it. To avoid false
//! positives, we should filter out any modifications that would produce a valid
//! transition.
//!
//! We can do this by generating arbitrary:
//! - storage modifications for any of the storage modifications performed by a
//!   transition
//! - any other storage key used by PoS
//! - any other storage key not used by PoS
//!
//! A modification for integer based values can be addition or subtraction their
//! value.

#[cfg(test)]
mod tests {

    use anoma::ledger::pos::anoma_proof_of_stake::PosBase;
    use anoma::ledger::pos::PosParams;
    use anoma::types::storage::Epoch;
    use anoma::types::token;
    use anoma_vm_env::proof_of_stake::parameters::testing::arb_pos_params;
    use anoma_vm_env::proof_of_stake::{staking_token_address, PosVP};
    use anoma_vm_env::tx_prelude::Address;
    use proptest::prelude::*;
    use proptest::prop_state_machine;
    use proptest::state_machine::{AbstractStateMachine, StateMachineTest};
    use proptest::test_runner::Config;
    use test_log::test;

    use super::testing::{
        arb_valid_pos_action, PosStorageChanges, ValidPosAction,
    };
    use crate::native_vp::TestNativeVpEnv;
    use crate::tx::{tx_host_env, TestTxEnv};

    prop_state_machine! {
        #![proptest_config(Config {
            // Instead of the default 256, we only run 10 because otherwise it
            // takes too long
            cases: 10,
            .. Config::default()
        })]
        #[test]
        /// A `StateMachineTest` implemented on `PosState`
        fn pos_vp_state_machine_test(sequential 1..20 => ConcretePosState);
    }

    /// Abstract representation of a state of PoS system
    #[derive(Clone, Debug)]
    struct AbstractPosState {
        /// Current epoch
        epoch: Epoch,
        /// Parameters
        params: PosParams,
        /// Valid PoS changes in the current transaction
        valid_actions: Vec<ValidPosAction>,
        /// Assuming to be empty in the initial state in
        /// `StateMachineTest::init_test`
        invalid_pos_changes: Vec<InvalidPosChange>,
        /// Empty in the initial state in `StateMachineTest::init_test`
        invalid_arbitrary_changes: crate::storage::Changes,
        /// Valid PoS changes committed to storage
        committed_valid_actions: Vec<ValidPosAction>,
    }

    /// The PoS system under test
    #[derive(Debug)]
    struct ConcretePosState;

    /// State machine transitions
    #[derive(Clone, Debug)]
    enum Transition {
        /// Commit all the tx changes already applied in the tx env
        CommitTx,
        /// Switch to a new epoch. This will also commit all the applied valid
        /// transactions.
        NextEpoch,
        /// Valid changes use the current epoch to apply changes correctly
        Valid(ValidPosAction),
        /// Invalid changes with valid data structures
        InvalidPos(InvalidPosChange),
        /// Invalid changes with arbitrary data
        InvalidArbitrary(crate::storage::Change),
    }

    #[derive(Clone, Debug)]
    struct InvalidPosChange {
        changes: PosStorageChanges,
        /// Invalid changes can be applied at arbitrary epoch
        epoch: Epoch,
    }

    impl StateMachineTest for ConcretePosState {
        type Abstract = AbstractPosState;
        type ConcreteState = Self;

        fn init_test(
            initial_state: <Self::Abstract as AbstractStateMachine>::State,
        ) -> Self::ConcreteState {
            println!();
            println!("New test case");

            // Initialize the transaction env
            let mut tx_env = TestTxEnv::default();

            // Set the epoch
            let storage = &mut tx_env.storage;
            storage.block.epoch = initial_state.epoch;

            // Initialize PoS storage
            storage
                .init_genesis(
                    &initial_state.params,
                    [].into_iter(),
                    initial_state.epoch,
                )
                .unwrap();

            // Make sure that the staking token account exist
            tx_env.spawn_accounts([staking_token_address()]);

            // Use the `tx_env` for host env calls
            tx_host_env::set(tx_env);

            for change in initial_state.committed_valid_actions {
                println!("Apply init state change {:#?}", change);
                change.apply()
            }
            Self
        }

        fn apply_concrete(
            test_state: Self::ConcreteState,
            transition: <Self::Abstract as AbstractStateMachine>::Transition,
        ) -> Self::ConcreteState {
            match transition {
                Transition::CommitTx => {
                    tx_host_env::commit_tx_and_block();
                }
                Transition::NextEpoch => {
                    tx_host_env::with(|env| {
                        // Also commit the last transaction(s) changes
                        env.commit_tx_and_block();

                        env.storage.block.epoch =
                            env.storage.block.epoch.next();
                    });
                }
                Transition::Valid(change) => {
                    change.apply();

                    // Use the tx_env to run PoS VP
                    let tx_env = tx_host_env::take();
                    let vp_env = TestNativeVpEnv::new(tx_env);
                    let result = vp_env.validate_tx(PosVP::new, |_tx_data| {});
                    // Put the tx_env back before checking conditions
                    tx_host_env::set(vp_env.tx_env);

                    // Post-condition: Changes must be accepted
                    // TODO this also depends on the current state
                    let result = result
                        .expect("Validation of valid changes must not fail!");
                    assert!(result, "Validation of valid changes must pass!");
                }
                Transition::InvalidPos(_) => todo!(),
                Transition::InvalidArbitrary(_) => todo!(),
            }

            test_state
        }

        fn test_sequential(
            initial_state: <Self::Abstract as AbstractStateMachine>::State,
            transitions: Vec<
                <Self::Abstract as AbstractStateMachine>::Transition,
            >,
        ) {
            let mut state = Self::init_test(initial_state);
            println!("Transitions {}", transitions.len());
            for (i, transition) in transitions.into_iter().enumerate() {
                println!("Apply transition {}: {:#?}", i, transition);
                state = Self::apply_concrete(state, transition);
                Self::invariants(&state);
            }
        }
    }

    impl AbstractStateMachine for AbstractPosState {
        type State = Self;
        type Transition = Transition;

        fn init_state() -> BoxedStrategy<Self::State> {
            (arb_pos_params(), 0..100_u64)
                .prop_flat_map(|(params, epoch)| {
                    // We're starting from an empty state
                    let state = vec![];
                    let epoch = Epoch(epoch);
                    arb_valid_pos_action(&params, state).prop_map(
                        move |valid_action| Self {
                            epoch,
                            params: params.clone(),
                            valid_actions: vec![],
                            invalid_pos_changes: vec![],
                            invalid_arbitrary_changes: vec![],
                            committed_valid_actions: vec![valid_action],
                        },
                    )
                })
                .boxed()
        }

        fn transitions(state: &Self::State) -> BoxedStrategy<Self::Transition> {
            let valid_actions = state.all_valid_actions();
            prop_oneof![
                Just(Transition::CommitTx),
                Just(Transition::NextEpoch),
                arb_valid_pos_action(&state.params, valid_actions)
                    .prop_map(Transition::Valid)
            ]
            .boxed()
        }

        fn apply_abstract(
            mut state: Self::State,
            transition: &Self::Transition,
        ) -> Self::State {
            match transition {
                Transition::CommitTx => {
                    state.commit_tx();
                }
                Transition::NextEpoch => {
                    state.commit_tx();
                    state.epoch = state.epoch.next();
                }
                Transition::Valid(transition) => {
                    state.valid_actions.push(transition.clone());
                }
                Transition::InvalidPos(change) => {
                    state.invalid_pos_changes.push(change.clone());
                }
                Transition::InvalidArbitrary(transition) => {
                    state.invalid_arbitrary_changes.push(transition.clone());
                }
            }
            state
        }

        fn preconditions(
            state: &Self::State,
            transition: &Self::Transition,
        ) -> bool {
            match transition {
                Transition::CommitTx => true,
                Transition::NextEpoch => true,
                Transition::Valid(action) => match action {
                    ValidPosAction::InitValidator(_) => true,
                    ValidPosAction::Bond {
                        amount,
                        owner,
                        validator,
                    } => state.is_validator(validator),
                    ValidPosAction::Unbond {
                        amount,
                        owner,
                        validator,
                    } => {
                        state.is_validator(validator)
                            && state.has_enough_bonds(owner, validator, *amount)
                    }
                    ValidPosAction::Withdraw { owner, validator } => {
                        state.is_validator(validator)
                            && state.has_withdrawable_unbonds(owner, validator)
                    }
                },
                Transition::InvalidPos(_) => true,
                Transition::InvalidArbitrary(_) => true,
            }
        }
    }

    impl ConcretePosState {}

    impl AbstractPosState {
        /// Commit a transaction. This will append the `valid_actions` to the
        /// `committed_valid_actions` and discard invalid changes.
        fn commit_tx(&mut self) {
            let committed_valid_actions =
                std::mem::take(&mut self.valid_actions);
            self.committed_valid_actions
                .extend(committed_valid_actions.into_iter());
            self.invalid_pos_changes = vec![];
            self.invalid_arbitrary_changes = vec![];
        }

        /// Get all the valid actions since genesis
        fn all_valid_actions(&self) -> Vec<ValidPosAction> {
            [
                self.committed_valid_actions.clone(),
                self.valid_actions.clone(),
            ]
            .concat()
        }

        /// Find if the given address is a validator
        fn is_validator(&self, addr: &Address) -> bool {
            self.all_valid_actions().iter().any(|action| match action {
                ValidPosAction::InitValidator(validator) => validator == addr,
                _ => false,
            })
        }

        /// Find if the given owner and validator has bonds that are greater or
        /// equal to the given amount, so that it can be unbonded.
        /// Note that for self-bonds, `owner == validator`.
        fn has_enough_bonds(
            &self,
            owner: &Address,
            validator: &Address,
            amount: token::Amount,
        ) -> bool {
            let raw_amount: u64 = amount.into();
            let mut total_bonds: u64 = 0;
            for action in self.all_valid_actions().into_iter() {
                match action {
                    ValidPosAction::Bond {
                        amount,
                        owner: bond_owner,
                        validator: bond_validator,
                    } => {
                        if owner == &bond_owner && validator == &bond_validator
                        {
                            let raw_amount: u64 = amount.into();
                            total_bonds += raw_amount;
                        }
                    }
                    ValidPosAction::Unbond {
                        amount,
                        owner: bond_owner,
                        validator: bond_validator,
                    } => {
                        if owner == &bond_owner && validator == &bond_validator
                        {
                            let raw_amount: u64 = amount.into();
                            total_bonds -= raw_amount;
                        }
                    }
                    ValidPosAction::InitValidator(_) => {}
                    ValidPosAction::Withdraw { owner, validator } => {}
                }
            }

            total_bonds >= raw_amount
        }

        /// Find if the given owner and validator has unbonds that are ready to
        /// be withdrawn.
        /// Note that for self-bonds, `owner == validator`.
        fn has_withdrawable_unbonds(
            &self,
            owner: &Address,
            validator: &Address,
        ) -> bool {
            // TODO
            true
        }
    }
}

/// Testing helpers
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use std::collections::HashMap;

    use anoma::ledger::pos::anoma_proof_of_stake::btree_set::BTreeSetShims;
    use anoma::types::key::ed25519::{self, PublicKey};
    use anoma::types::storage::Epoch;
    use anoma::types::{address, token};
    use anoma_vm_env::proof_of_stake::epoched::{
        DynEpochOffset, Epoched, EpochedDelta,
    };
    use anoma_vm_env::proof_of_stake::types::{
        Bond, ValidatorState, VotingPower, VotingPowerDelta, WeightedValidator,
    };
    use anoma_vm_env::proof_of_stake::{
        staking_token_address, BondId, Bonds, PosParams,
    };
    use anoma_vm_env::tx_prelude::{Address, PoS};
    use derivative::Derivative;
    use proptest::prelude::*;

    use crate::tx::tx_host_env;

    #[derive(Clone, Debug, Default)]
    pub struct TestValidator {
        pub address: Option<Address>,
        pub staking_reward_address: Option<Address>,
        pub stake: Option<token::Amount>,
        /// Balance is a pair of token address and its amount
        pub unstaked_balances: Vec<(Address, token::Amount)>,
    }

    /// The state of PoS storage is made of changes, which must be applied in
    /// the same order to get to the current state.
    pub type PosStorageChanges = Vec<PosStorageChange>;

    #[derive(Clone, Debug)]
    pub enum ValidPosAction {
        InitValidator(Address),
        Bond {
            amount: token::Amount,
            owner: Address,
            validator: Address,
        },
        Unbond {
            amount: token::Amount,
            owner: Address,
            validator: Address,
        },
        Withdraw {
            owner: Address,
            validator: Address,
        },
    }

    /// PoS storage modifications
    #[derive(Clone, Derivative)]
    #[derivative(Debug)]

    pub enum PosStorageChange {
        /// Ensure that the account exists when initialing a valid new
        /// validator or delegation from a new owner
        SpawnAccount {
            address: Address,
        },
        Bond {
            owner: Address,
            validator: Address,
            delta: i128,
        },
        Unbond {
            owner: Address,
            validator: Address,
            delta: i128,
        },
        TotalVotingPower {
            vp_delta: i128,
            offset: DynEpochOffset,
        },
        ValidatorSet {
            // change: ValidatorSetChange,
            validator: Address,
            token_delta: i128,
            offset: DynEpochOffset,
        },
        ValidatorConsensusKey {
            validator: Address,
            #[derivative(Debug = "ignore")]
            pk: PublicKey,
        },
        ValidatorStakingRewardsAddress {
            validator: Address,
            address: Address,
        },
        ValidatorTotalDeltas {
            validator: Address,
            delta: i128,
            offset: DynEpochOffset,
        },
        ValidatorVotingPower {
            validator: Address,
            vp_delta: i64,
            offset: DynEpochOffset,
        },
        ValidatorState {
            validator: Address,
            state: ValidatorState,
        },
        StakingTokenPosBalance {
            delta: i128,
        },
        ValidatorAddressRawHash {
            address: Address,
        },
    }

    /// PoS validator set storage modifications
    // TODO these are too specific for state machine tests, because they depend
    // on the state in each epoch when updating them, more abstract will
    // do nicely!
    #[derive(Clone, Debug)]
    pub enum ValidatorSetChange {
        AddActive {
            validator: Address,
            voting_power: u64,
        },
        RmActive {
            validator: Address,
            voting_power: u64,
        },
        UpdateActive {
            validator: Address,
            voting_power: u64,
            delta: i128,
        },
        AddInactive {
            validator: Address,
            voting_power: u64,
        },
        RmInactive {
            validator: Address,
            voting_power: u64,
        },
        UpdateInactive {
            validator: Address,
            voting_power: u64,
            delta: i128,
        },
    }

    pub fn arb_valid_pos_action(
        _params: &PosParams,
        valid_actions: Vec<ValidPosAction>,
    ) -> impl Strategy<Value = ValidPosAction> {
        let validators: Vec<Address> = valid_actions
            .iter()
            .filter_map(|action| match action {
                ValidPosAction::InitValidator(addr) => Some(addr.clone()),
                _ => None,
            })
            .collect();
        let init_validator = address::testing::arb_established_address()
            .prop_map(|addr| {
                ValidPosAction::InitValidator(Address::Established(addr))
            });
        if validators.is_empty() {
            init_validator.boxed()
        } else {
            let arb_validator = proptest::sample::select(validators);
            let arb_validator_or_address = prop_oneof![
                arb_validator.clone(),
                address::testing::arb_established_address()
                    .prop_map(Address::Established),
            ];
            // TODO add other transitions
            prop_oneof![
                init_validator,
                (
                    // We select bond amount safely below the `u64::MAX` to
                    // avoid their sums overflowing `u64` in the model
                    (0..dbg!(u64::MAX / 1_000)),
                    arb_validator_or_address,
                    arb_validator
                )
                    .prop_map(|(amount, owner, validator)| {
                        ValidPosAction::Bond {
                            amount: amount.into(),
                            owner,
                            validator,
                        }
                    }),
            ]
            .boxed()
        }
    }

    impl ValidPosAction {
        /// Apply a valid PoS storage action. This will use the current epoch
        /// from the tx env to apply the change on epoched data as expected by
        /// the VP.
        pub fn apply(self) {
            // We'll read the current state to find what storage changes must be
            // applied
            use anoma_vm_env::tx_prelude::PosRead;
            let params = PoS.read_pos_params();

            let current_epoch = tx_host_env::with(|env| {
                // Reset the gas meter on each change, so that we never run
                // out in this test
                env.gas_meter.reset();
                env.storage.block.epoch
            });
            println!("Current epoch {}", current_epoch);

            let changes = self.into_storage_changes(&params, current_epoch);
            for change in changes {
                apply_valid_pos_storage_change(change, &params, current_epoch)
            }
        }

        /// Convert a valid PoS action to PoS storage changes
        pub fn into_storage_changes(
            self,
            params: &PosParams,
            current_epoch: Epoch,
        ) -> PosStorageChanges {
            use anoma_vm_env::tx_prelude::PosRead;

            match self {
                ValidPosAction::InitValidator(addr) => {
                    let offset = DynEpochOffset::PipelineLen;
                    vec![
                        PosStorageChange::SpawnAccount {
                            address: addr.clone(),
                        },
                        PosStorageChange::ValidatorAddressRawHash {
                            address: addr.clone(),
                        },
                        PosStorageChange::ValidatorSet {
                            // change: if has_vacant_active_validator_slots(
                            //     params,
                            //     current_epoch,
                            // ) {
                            //     ValidatorSetChange::AddActive {
                            //         validator: addr.clone(),
                            //         voting_power: 0,
                            //     }
                            // } else {
                            //     ValidatorSetChange::AddInactive {
                            //         validator: addr.clone(),
                            //         voting_power: 0,
                            //     }
                            // },
                            validator: addr.clone(),
                            token_delta: 0,
                            offset,
                        },
                        PosStorageChange::ValidatorConsensusKey {
                            validator: addr.clone(),
                            pk: ed25519::testing::keypair_1().public,
                        },
                        PosStorageChange::ValidatorStakingRewardsAddress {
                            validator: addr.clone(),
                            address: address::testing::established_address_1(),
                        },
                        PosStorageChange::ValidatorState {
                            validator: addr.clone(),
                            state: ValidatorState::Pending,
                        },
                        PosStorageChange::ValidatorState {
                            validator: addr.clone(),
                            state: ValidatorState::Candidate,
                        },
                        PosStorageChange::ValidatorTotalDeltas {
                            validator: addr.clone(),
                            delta: 0,
                            offset,
                        },
                        PosStorageChange::ValidatorVotingPower {
                            validator: addr,
                            vp_delta: 0,
                            offset,
                        },
                    ]
                }
                ValidPosAction::Bond {
                    amount,
                    owner,
                    validator,
                } => {
                    let offset = DynEpochOffset::PipelineLen;
                    // We first need to find if the validator's voting power
                    // is affected
                    let token_delta = amount.change();
                    let validator_total_deltas =
                        PoS.read_validator_total_deltas(&validator).unwrap();
                    let total_delta = validator_total_deltas
                        .get_at_offset(
                            current_epoch,
                            DynEpochOffset::PipelineLen,
                            params,
                        )
                        .unwrap_or_default();
                    // We convert the tokens from micro units to whole tokens
                    // with division by 10^6
                    let vp_before =
                        params.votes_per_token * (total_delta / 1_000_000);
                    let vp_after = params.votes_per_token
                        * ((total_delta + token_delta) / 1_000_000);
                    let is_voting_power_changed = vp_before != vp_after;

                    let mut changes = Vec::with_capacity(7);
                    // ensure that the owner account exists
                    changes.push(PosStorageChange::SpawnAccount {
                        address: owner.clone(),
                    });

                    // If the bond increases the voting power, more storage
                    // updates are needed
                    if dbg!(is_voting_power_changed) {
                        // voting power delta
                        let vp_delta = vp_after - vp_before;

                        // IMPORTANT: we have to update `ValidatorSet` and
                        // `TotalVotingPower` before we update
                        // `ValidatorTotalDeltas`, because they needs to
                        // read the total deltas before they change.
                        changes.extend([
                            PosStorageChange::ValidatorSet {
                                validator: validator.clone(),
                                token_delta,
                                offset,
                            },
                            PosStorageChange::TotalVotingPower {
                                vp_delta,
                                offset,
                            },
                            PosStorageChange::ValidatorVotingPower {
                                validator: validator.clone(),
                                vp_delta: vp_delta.try_into().unwrap(),
                                offset,
                            },
                        ]);
                    }

                    changes.extend([
                        PosStorageChange::Bond {
                            owner,
                            validator: validator.clone(),
                            delta: token_delta,
                        },
                        PosStorageChange::ValidatorTotalDeltas {
                            validator,
                            delta: token_delta,
                            offset,
                        },
                        PosStorageChange::StakingTokenPosBalance {
                            delta: token_delta,
                        },
                    ]);

                    // let validator_sets = PoS.read_validator_set();
                    // let validator_set = validator_sets
                    //     .get_at_offset(
                    //         current_epoch,
                    //         DynEpochOffset::PipelineLen,
                    //         params,
                    //     )
                    //     .unwrap();
                    // let staked_tokens_pre: u64 =
                    //     total_delta.try_into().unwrap();
                    // let amount: u64 = amount.into();
                    // let staked_tokens_post: u64 =
                    //     staked_tokens_pre + amount;
                    // let voting_power_pre =
                    //     VotingPower::from_tokens(staked_tokens_pre,
                    // params);
                    // let voting_power_post = VotingPower::from_tokens(
                    //     staked_tokens_post,
                    //     params,
                    // );
                    // let weighted_validator_pre = WeightedValidator {
                    //     voting_power: voting_power_pre,
                    //     address: validator.clone(),
                    // };
                    // // TODO this should work for unbond too
                    // if validator_set
                    //     .active
                    //     .contains(&weighted_validator_pre)
                    // {
                    //     match validator_set.inactive.last_shim() {
                    //         Some(max_inactive) => {
                    //             if max_inactive.voting_power
                    //                 <= voting_power_post
                    //             {
                    //                 if validator_set
                    //                     .active
                    //
                    // .contains(&weighted_validator_pre)
                    //                 {
                    //                     changes.extend([

                    //             PosStorageChange::ValidatorSet {
                    //                 change:
                    //                     ValidatorSetChange::UpdateActive
                    // {     validator:
                    // validator.clone(),
                    //                         voting_power:
                    // voting_power_pre.into(),
                    //                         delta,
                    //                     },
                    //     offset,
                    //             },
                    //                 ])
                    //                 } else {
                    //                     changes.extend([

                    //             PosStorageChange::ValidatorSet {
                    //                 change:
                    //
                    // ValidatorSetChange::UpdateInactive {
                    //     validator: validator.clone(),
                    //                         voting_power:
                    // voting_power_pre.into(),
                    //                         delta,
                    //                     },
                    //     offset,
                    //             },
                    //                 ])
                    //                 }
                    //             } else {
                    //                 changes.extend([

                    //             PosStorageChange::ValidatorSet {
                    //                 change:
                    //                     ValidatorSetChange::RmActive {
                    //     validator: validator.clone(),
                    //                         voting_power:
                    // voting_power_pre.into(),
                    //                     },
                    //     offset,
                    //             },
                    //             PosStorageChange::ValidatorSet {
                    //                 change:
                    //                     ValidatorSetChange::AddInactive {
                    //     validator: validator.clone(),
                    //                         voting_power:
                    // voting_power_post.into(),
                    //                     },
                    //     offset,
                    //             },
                    //             PosStorageChange::ValidatorSet {
                    //                 change:
                    //                     ValidatorSetChange::RmInactive {
                    //                         validator:
                    // max_inactive.address.clone(),
                    //                         voting_power:
                    // max_inactive.voting_power.into(),
                    //                     },
                    //     offset,
                    //             },
                    //             PosStorageChange::ValidatorSet {
                    //                 change:
                    //                     ValidatorSetChange::AddActive {
                    //                         validator:
                    // max_inactive.address.clone(),
                    //                         voting_power:
                    // max_inactive.voting_power.into(),
                    //                     },
                    //     offset,
                    //             },
                    //                 ])
                    //             }
                    //         }
                    //         None => changes.extend([
                    //             PosStorageChange::ValidatorSet {
                    //                 change:
                    //                     ValidatorSetChange::UpdateActive
                    // {
                    // validator: validator.clone(),
                    //                         voting_power:
                    // voting_power_pre
                    // .into(),
                    // delta,                     },
                    //                 offset,
                    //             },
                    //         ]),
                    //     }
                    // }

                    changes
                }
                ValidPosAction::Unbond {
                    amount,
                    owner,
                    validator,
                } => todo!(),
                ValidPosAction::Withdraw { owner, validator } => todo!(),
            }
        }
    }

    pub fn apply_valid_pos_storage_change(
        change: PosStorageChange,
        params: &PosParams,
        current_epoch: Epoch,
    ) {
        use anoma_vm_env::tx_prelude::{PosRead, PosWrite};

        match dbg!(change) {
            PosStorageChange::SpawnAccount { address } => {
                tx_host_env::with(move |env| {
                    env.spawn_accounts([&address]);
                });
            }
            PosStorageChange::Bond {
                owner,
                validator,
                delta,
            } => {
                let bond_id = BondId {
                    source: owner,
                    validator,
                };
                let bonds = dbg!(PoS.read_bond(&bond_id));
                let amount: u64 = delta.try_into().unwrap();
                let amount: token::Amount = amount.into();
                let mut value = Bond {
                    deltas: HashMap::default(),
                };
                value.deltas.insert(
                    (current_epoch + DynEpochOffset::PipelineLen.value(params))
                        .into(),
                    amount,
                );
                let bonds = match bonds {
                    Some(mut bonds) => {
                        bonds.add(value, current_epoch, params);
                        bonds
                    }
                    None => Bonds::init(value, current_epoch, params),
                };
                PoS.write_bond(&bond_id, dbg!(bonds));
            }
            PosStorageChange::Unbond {
                owner: _,
                validator: _,
                delta: _,
            } => todo!(),
            PosStorageChange::TotalVotingPower { vp_delta, offset } => {
                let mut total_voting_powers = PoS.read_total_voting_power();
                let vp_delta: i64 = vp_delta.try_into().unwrap();
                total_voting_powers.add_at_offset(
                    VotingPowerDelta::from(vp_delta),
                    current_epoch,
                    offset,
                    params,
                );
                PoS.write_total_voting_power(total_voting_powers)
            }
            PosStorageChange::ValidatorAddressRawHash { address } => {
                PoS.write_validator_address_raw_hash(&address);
            }
            PosStorageChange::ValidatorSet {
                validator,
                token_delta,
                offset,
            } => {
                apply_validator_set_change(
                    validator,
                    token_delta,
                    offset,
                    current_epoch,
                    params,
                );
            }
            PosStorageChange::ValidatorConsensusKey { validator, pk } => {
                let consensus_key = PoS
                    .read_validator_consensus_key(&validator)
                    .map(|mut consensus_keys| {
                        consensus_keys.set(pk.clone(), current_epoch, params);
                        consensus_keys
                    })
                    .unwrap_or_else(|| {
                        Epoched::init(pk, current_epoch, params)
                    });
                PoS.write_validator_consensus_key(&validator, consensus_key);
            }
            PosStorageChange::ValidatorStakingRewardsAddress {
                validator,
                address,
            } => {
                PoS.write_validator_staking_reward_address(&validator, address);
            }
            PosStorageChange::ValidatorTotalDeltas {
                validator,
                delta,
                offset,
            } => {
                let total_deltas = PoS
                    .read_validator_total_deltas(&validator)
                    .map(|mut total_deltas| {
                        total_deltas.add_at_offset(
                            delta,
                            current_epoch,
                            offset,
                            params,
                        );
                        dbg!(total_deltas)
                    })
                    .unwrap_or_else(|| {
                        dbg!(EpochedDelta::init_at_offset(
                            delta,
                            current_epoch,
                            DynEpochOffset::PipelineLen,
                            params,
                        ))
                    });
                PoS.write_validator_total_deltas(&validator, total_deltas);
            }
            PosStorageChange::ValidatorVotingPower {
                validator,
                vp_delta: delta,
                offset,
            } => {
                let voting_power = PoS
                    .read_validator_voting_power(&validator)
                    .map(|mut voting_powers| {
                        voting_powers.add_at_offset(
                            delta.into(),
                            current_epoch,
                            offset,
                            params,
                        );
                        voting_powers
                    })
                    .unwrap_or_else(|| {
                        EpochedDelta::init_at_offset(
                            delta.into(),
                            current_epoch,
                            DynEpochOffset::PipelineLen,
                            params,
                        )
                    });
                PoS.write_validator_voting_power(&validator, voting_power);
            }
            PosStorageChange::ValidatorState { validator, state } => {
                let state = PoS
                    .read_validator_state(&validator)
                    .map(|mut states| {
                        states.set(state, current_epoch, params);
                        states
                    })
                    .unwrap_or_else(|| {
                        Epoched::init_at_genesis(state, current_epoch)
                    });
                PoS.write_validator_state(&validator, state);
            }
            PosStorageChange::StakingTokenPosBalance { delta } => {
                let balance_key = token::balance_key(
                    &staking_token_address(),
                    &<PoS as PosRead>::POS_ADDRESS,
                )
                .to_string();
                let mut balance: token::Amount =
                    tx_host_env::read(&balance_key).unwrap_or_default();
                if delta < 0 {
                    let to_spend: u64 = (-delta).try_into().unwrap();
                    let to_spend: token::Amount = to_spend.into();
                    balance.spend(&to_spend);
                } else {
                    let to_recv: u64 = delta.try_into().unwrap();
                    let to_recv: token::Amount = to_recv.into();
                    balance.receive(&to_recv);
                }
                tx_host_env::write(&balance_key, balance);
            }
        }
    }

    pub fn apply_validator_set_change(
        // change: ValidatorSetChange,
        validator: Address,
        token_delta: i128,
        offset: DynEpochOffset,
        current_epoch: Epoch,
        params: &PosParams,
    ) {
        use anoma_vm_env::tx_prelude::{PosRead, PosWrite};

        let validator_total_deltas =
            PoS.read_validator_total_deltas(&validator);
        // println!("Read validator set");
        let mut validator_set = PoS.read_validator_set();
        // println!("Read validator set: {:#?}", validator_set);
        validator_set.update_from_offset(
            |validator_set, epoch| {
                let total_delta = validator_total_deltas
                    .as_ref()
                    .and_then(|delta| delta.get(epoch));
                match total_delta {
                    Some(total_delta) => {
                        let tokens_pre: u64 = total_delta.try_into().unwrap();
                        let voting_power_pre =
                            VotingPower::from_tokens(tokens_pre, params);
                        let tokens_post: u64 =
                            dbg!(total_delta + dbg!(token_delta))
                                .try_into()
                                .unwrap();
                        let voting_power_post =
                            VotingPower::from_tokens(tokens_post, params);
                        let weighed_validator_pre = WeightedValidator {
                            voting_power: voting_power_pre,
                            address: validator.clone(),
                        };
                        let weighed_validator_post = WeightedValidator {
                            voting_power: voting_power_post,
                            address: validator.clone(),
                        };
                        if validator_set.active.contains(&weighed_validator_pre)
                        {
                            let max_inactive_validator =
                                validator_set.inactive.last_shim();
                            let max_voting_power = max_inactive_validator
                                .map(|v| v.voting_power)
                                .unwrap_or_default();
                            if voting_power_post < max_voting_power {
                                let activate_max =
                                    validator_set.inactive.pop_last_shim();
                                let popped = validator_set
                                    .active
                                    .remove(&weighed_validator_pre);
                                validator_set
                                    .inactive
                                    .insert(weighed_validator_pre);
                                if let Some(activate_max) = activate_max {
                                    validator_set.active.insert(activate_max);
                                }
                            } else {
                                validator_set
                                    .active
                                    .remove(&weighed_validator_pre);
                                validator_set
                                    .active
                                    .insert(weighed_validator_post);
                            }
                        } else {
                            let min_active_validator =
                                validator_set.active.first_shim();
                            let min_voting_power = min_active_validator
                                .map(|v| v.voting_power)
                                .unwrap_or_default();
                            if voting_power_post > min_voting_power {
                                let deactivate_min =
                                    validator_set.active.pop_first_shim();
                                let popped = validator_set
                                    .inactive
                                    .remove(&weighed_validator_pre);
                                debug_assert!(popped);
                                validator_set
                                    .active
                                    .insert(weighed_validator_post);
                                if let Some(deactivate_min) = deactivate_min {
                                    validator_set
                                        .inactive
                                        .insert(deactivate_min);
                                }
                            } else {
                                validator_set
                                    .inactive
                                    .remove(&weighed_validator_pre);
                                validator_set
                                    .inactive
                                    .insert(weighed_validator_post);
                            }
                        }
                    }
                    None => {
                        let tokens: u64 = token_delta.try_into().unwrap();
                        let weighed_validator = WeightedValidator {
                            voting_power: VotingPower::from_tokens(
                                tokens, params,
                            ),
                            address: validator.clone(),
                        };
                        if has_vacant_active_validator_slots(
                            params,
                            current_epoch,
                        ) {
                            validator_set.active.insert(weighed_validator);
                        } else {
                            validator_set.inactive.insert(weighed_validator);
                        }
                    }
                }
            },
            current_epoch,
            offset,
            params,
        );
        // println!("Write validator set {:#?}", validator_set);
        PoS.write_validator_set(validator_set);

        // match change {
        //     ValidatorSetChange::AddActive {
        //         validator,
        //         voting_power,
        //     } => {
        //         // println!("Read validator set");
        //         let mut validator_set = PoS.read_validator_set();
        //         // println!("Read validator set: {:#?}", validator_set);
        //         validator_set.update_from_offset(
        //             |validator_set, _epoch| {
        //                 let validator = WeightedValidator {
        //                     // TODO this depends on the epoch
        //                     voting_power: VotingPower::from(voting_power),
        //                     address: validator.clone(),
        //                 };
        //                 validator_set.active.insert(validator);
        //             },
        //             epoch,
        //             offset,
        //             params,
        //         );
        //         // println!("Write validator set {:#?}", validator_set);
        //         PoS.write_validator_set(validator_set);
        //     }
        //     ValidatorSetChange::AddInactive {
        //         validator,
        //         voting_power,
        //     } => {
        //         // println!("Read validator set");
        //         let mut validator_set = PoS.read_validator_set();
        //         // println!("Read validator set: {:#?}", validator_set);
        //         validator_set.update_from_offset(
        //             |validator_set, _epoch| {
        //                 let validator = WeightedValidator {
        //                     // TODO this depends on the epoch
        //                     voting_power: VotingPower::from(voting_power),
        //                     address: validator.clone(),
        //                 };
        //                 validator_set.inactive.insert(validator);
        //             },
        //             epoch,
        //             offset,
        //             params,
        //         );
        //         // println!("Write validator set {:#?}", validator_set);
        //         PoS.write_validator_set(validator_set);
        //     }
        //     ValidatorSetChange::RmActive {
        //         validator,
        //         voting_power,
        //     } => {
        //         // println!("Read validator set");
        //         let mut validator_set = PoS.read_validator_set();
        //         // println!("Read validator set: {:#?}", validator_set);
        //         validator_set.update_from_offset(
        //             |validator_set, _epoch| {
        //                 let validator = WeightedValidator {
        //                     // TODO this depends on the epoch
        //                     voting_power: VotingPower::from(voting_power),
        //                     address: validator.clone(),
        //                 };
        //                 if !validator_set.active.remove(&validator) {
        //                     panic!(
        //                         "Active validator {:#?} not found in {:#?}",
        //                         validator, validator_set.inactive
        //                     )
        //                 }
        //             },
        //             epoch,
        //             offset,
        //             params,
        //         );
        //         // println!("Write validator set {:#?}", validator_set);
        //         PoS.write_validator_set(validator_set);
        //     }
        //     ValidatorSetChange::UpdateActive {
        //         validator,
        //         voting_power,
        //         delta,
        //     } => {
        //         // println!("Read validator set");
        //         let mut validator_set = PoS.read_validator_set();
        //         // println!("Read validator set: {:#?}", validator_set);
        //         validator_set.update_from_offset(
        //             |validator_set, _epoch| {
        //                 let validator_pre = WeightedValidator {
        //                     // TODO this depends on the epoch
        //                     voting_power: VotingPower::from(voting_power),
        //                     address: validator.clone(),
        //                 };
        //                 let voting_power_post: u64 =
        //                     (voting_power as i128 +
        // delta).try_into().unwrap();                 let
        // validator_post = WeightedValidator {                     //
        // TODO this depends on the epoch
        // voting_power: VotingPower::from(voting_power_post),
        //                     address: validator.clone(),
        //                 };
        //                 if !validator_set.active.remove(&validator_pre) {
        //                     panic!(
        //                         "Inactive validator {:#?} not found in
        // {:#?}",                         validator,
        // validator_set.inactive                     );
        //                 }
        //                 validator_set.active.insert(validator_post);
        //             },
        //             epoch,
        //             offset,
        //             params,
        //         );
        //         // println!("Write validator set {:#?}", validator_set);
        //         PoS.write_validator_set(validator_set);
        //     }
        //     ValidatorSetChange::RmInactive {
        //         validator,
        //         voting_power,
        //     } => {
        //         // println!("Read validator set");
        //         let mut validator_set = PoS.read_validator_set();
        //         // println!("Read validator set: {:#?}", validator_set);
        //         validator_set.update_from_offset(
        //             |validator_set, _epoch| {
        //                 let validator = WeightedValidator {
        //                     // TODO this depends on the epoch
        //                     voting_power: VotingPower::from(voting_power),
        //                     address: validator.clone(),
        //                 };
        //                 if !validator_set.inactive.remove(&validator) {
        //                     panic!(
        //                         "Inactive validator {:#?} not found in
        // {:#?}",                         validator,
        // validator_set.inactive                     )
        //                 }
        //             },
        //             epoch,
        //             offset,
        //             params,
        //         );
        //         // println!("Write validator set {:#?}", validator_set);
        //         PoS.write_validator_set(validator_set);
        //     }
        //     ValidatorSetChange::UpdateInactive {
        //         validator: _,
        //         voting_power: _,
        //         delta: _,
        //     } => todo!(),
        // }
    }

    /// Find if there are any vacant active validator slots
    pub fn has_vacant_active_validator_slots(
        params: &PosParams,
        current_epoch: Epoch,
    ) -> bool {
        use anoma_vm_env::tx_prelude::PosRead;

        let validator_sets = PoS.read_validator_set();
        let validator_set = validator_sets
            .get_at_offset(current_epoch, DynEpochOffset::PipelineLen, params)
            .unwrap();
        params.max_validator_slots
            > validator_set.active.len().try_into().unwrap()
    }
}
