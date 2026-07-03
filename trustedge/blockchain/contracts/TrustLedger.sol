// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title TrustLedger
/// @notice On-chain record of PoA^2 block-finalization decisions (Sec. III-D,
/// Eq. 15). Each entry anchors the association-score snapshot that gated a
/// given federated-learning round's aggregation, giving the "immutable trust
/// ledger" described in Fig. 1's control-plane <-> blockchain-layer link.
contract TrustLedger {
    struct ConsensusRecord {
        uint256 round;
        bool authorityQuorumMet;
        bool associationThresholdMet;
        bool finalized;
        uint256 meanAssociationScoreScaled; // association score * 1e4, since Solidity has no floats
        bytes32 modelHash;
        uint256 timestamp;
    }

    mapping(uint256 => ConsensusRecord) public records;
    uint256[] public roundIndex;

    event BlockFinalized(uint256 indexed round, bytes32 modelHash, uint256 meanAssociationScoreScaled);
    event BlockRejected(uint256 indexed round, bool authorityQuorumMet, bool associationThresholdMet);

    /// @notice Records one round's PoA^2 finalization decision (Eq. 15).
    function recordConsensus(
        uint256 round,
        bool authorityQuorumMet,
        bool associationThresholdMet,
        uint256 meanAssociationScoreScaled,
        bytes32 modelHash
    ) public {
        bool finalized = authorityQuorumMet && associationThresholdMet;
        records[round] = ConsensusRecord(
            round, authorityQuorumMet, associationThresholdMet, finalized,
            meanAssociationScoreScaled, modelHash, block.timestamp
        );
        roundIndex.push(round);

        if (finalized) {
            emit BlockFinalized(round, modelHash, meanAssociationScoreScaled);
        } else {
            emit BlockRejected(round, authorityQuorumMet, associationThresholdMet);
        }
    }

    function getRecord(uint256 round) public view returns (ConsensusRecord memory) {
        return records[round];
    }

    function totalRounds() public view returns (uint256) {
        return roundIndex.length;
    }
}
