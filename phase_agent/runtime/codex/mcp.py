"""Small stdio MCP adapter for Codex takeover.

The adapter is deliberately thin: it exposes the existing Codex control
surface and never edits project state or calls a scientific tool directly.
Every proposal is sent through ``CodexOffline`` and therefore keeps the
Agent validation, approval, execution and memory lifecycle.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

from phase_agent.runtime.codex.offline import CodexOffline
from phase_agent.runtime.codex.team import TOOLS as TEAM_TOOLS, TOOL_NAMES, call_team_tool


_TOOLS = (
    {
        "name": "phase_regenerate_training_preflight",
        "description": "按用户明确授权，仅纳入精确绑定的已成功DFT并保留独立测试集，备份旧输入后使用原流程重新准备；不训练、不提交、不激活。",
        "inputSchema": {
            "type": "object",
            "required": [
                "task_id",
                "input_structure_sha256",
                "model_version",
                "expected_job_key",
                "independent_test_fraction",
                "seed",
                "user_instruction",
            ],
            "properties": {
                "task_id": {"type": "string"},
                "input_structure_sha256": {"type": "string"},
                "model_version": {"type": "string"},
                "expected_job_key": {"type": "string"},
                "independent_test_fraction": {"type": "number"},
                "seed": {"type": "integer"},
                "user_instruction": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_recover",
        "description": "回收和核验已登记结果。失败默认只返回步骤、批次、首错及续跑位置。",
        "inputSchema": {
            "type": "object",
            "properties": {"include_details": {"type": "boolean"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_recovery_diagnostic",
        "description": "只读单批次诊断。使用已登记批次 ID 或项目相对 submissions 目录；include_details=true 查看该批日志片段。",
        "inputSchema": {
            "type": "object",
            "required": ["batch_id"],
            "properties": {
                "batch_id": {"type": "string", "minLength": 1, "maxLength": 1024},
                "include_details": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_recover_phase_evidence",
        "description": "按用户明确指令核验并补登记冻结刷新相别证据；保留原始结果，不提交计算。",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "expected_model",
                "expected_hull",
                "epoch_root",
                "spec_path",
                "archive_path",
                "spec_sha256",
                "archive_sha256",
                "task_ids",
                "discard_task_ids",
                "user_instruction",
                "hull_source_policy",
            ],
            "properties": {
                **{
                    key: {"type": "string"}
                    for key in [
                        "expected_model",
                        "expected_hull",
                        "epoch_root",
                        "spec_path",
                        "archive_path",
                        "spec_sha256",
                        "archive_sha256",
                        "user_instruction",
                        "hull_source_policy",
                    ]
                },
                "task_ids": {"type": "array", "items": {"type": "string"}},
                "discard_task_ids": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "phase_publish_current_hull",
        "description": "按需读取当前相图并发布版本化图件；不提交计算。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_overview",
        "description": "读取或刷新绑定来源版本的精简项目目录页；不执行科学计算。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_status",
        "description": "读取完整状态和恢复证据；仅浏览目录先用 phase_overview。不回收、不迁移、不启动模型或计算。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_hpc_recovery",
        "description": "识别已登记的HPC启动/I/O故障，读取仅缺失项补算范围及原resume入口预检命令；只读，不提交。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_review_relax_mc_return",
        "description": "核对已登记Relax/MC包是否满足整批正式回收条件；quarantine=true按用户明确指令隔离旧版提前接纳的不完整批次，保留原结果和台账，不计算不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["bundle_id"],
            "properties": {"bundle_id": {"type": "string"}, "quarantine": {"type": "boolean"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_minimize_unsent_relax_mc_delivery",
        "description": "按用户明确授权核验并保守修订未提交Relax+MC输入包，保留旧包、任务、历史数据，不上传或提交。",
        "inputSchema": {
            "type": "object",
            "required": ["bundle_id", "source_sha256", "confirmed_unsubmitted"],
            "properties": {
                "bundle_id": {"type": "string"},
                "source_sha256": {"type": "string"},
                "confirmed_unsubmitted": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_register_relax_mc_input_attestation",
        "description": "核验并登记跨环境身份差异对应的远端输入文件哈希证据，不创建任务或提交。",
        "inputSchema": {
            "type": "object",
            "required": ["bundle_id", "attestation_path", "attestation_sha256"],
            "properties": {
                key: {"type": "string"}
                for key in ["bundle_id", "attestation_path", "attestation_sha256"]
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_preview_relax_mc_threshold_continuation",
        "description": "通过已登记阈值转换，隔离预检剩余 MC 配额；不写项目、不创建任务、不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["bundle_id"],
            "properties": {"bundle_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_register_relax_mc_threshold_transition",
        "description": "按用户明确指令核验并登记 Relax–MC 阶段阈值转换；保留原结果，不生成任务、不提交。",
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "bundle_id",
                "raw_path",
                "raw_sha256",
                "receipts_path",
                "receipts_sha256",
                "expected_review_digest",
                "user_instruction",
            ],
            "properties": {
                key: {"type": "string"}
                for key in [
                    "bundle_id",
                    "raw_path",
                    "raw_sha256",
                    "receipts_path",
                    "receipts_sha256",
                    "expected_review_digest",
                    "user_instruction",
                ]
            },
        },
    },
    {
        "name": "phase_set_remote_workspace",
        "description": "登记或修改当前项目的Linux远端工作根目录；仅本地元数据，不搬移、上传或提交。",
        "inputSchema": {
            "type": "object",
            "required": ["remote_root"],
            "properties": {"remote_root": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_propose_recovery",
        "description": "核对中断动作的登记产出并生成恢复审核方案，不重跑科学动作。",
        "inputSchema": {
            "type": "object",
            "required": [
                "invocation_id",
                "resolution",
                "evidence_note",
                "external_jobs",
                "output_inventory",
            ],
            "properties": {
                "invocation_id": {"type": "string"},
                "resolution": {"type": "string"},
                "evidence_note": {"type": "string"},
                "external_jobs": {"type": "string"},
                "output_inventory": {"type": "string"},
                "external_cost": {"type": ["number", "null"]},
                "staging_directory": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_name_direct_dft_handoff",
        "description": "按用户明确指令为已准备且未提交的 DIRECT DFT 单点交付安全命名，保留旧记录，不上传或提交。",
        "inputSchema": {
            "type": "object",
            "required": ["record_id", "source_sha256", "remote_root"],
            "properties": {
                "record_id": {"type": "string"},
                "source_sha256": {"type": "string"},
                "remote_root": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_migrate_project_dft_nelm",
        "description": "按用户明确指令迁移已确认的纯 DFT NELM 配置版本，保留历史任务，不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["expected_old", "expected_new", "value"],
            "properties": {
                "expected_old": {"type": "string"},
                "expected_new": {"type": "string"},
                "value": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_confirm_project_config",
        "description": "仅在用户明确确认配置更改时，通过现有配置审核和版本入口确认草稿。",
        "inputSchema": {
            "type": "object",
            "required": ["explicit"],
            "properties": {"explicit": {"type": "boolean"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_set_project_dft_nelm",
        "description": "按用户明确指令设置当前项目 DFT 单点电子步上限的配置草稿，不影响历史任务。",
        "inputSchema": {
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "integer"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_reconcile_recovery_refresh_epoch",
        "description": "按用户明确指令核验并登记已冻结恢复刷新模型的轮次；保留输入与回收原路径，不提交。",
        "inputSchema": {
            "type": "object",
            "required": [
                "model_version",
                "old_epoch",
                "new_epoch",
                "round_digest",
                "plan_checksum",
                "archive_sha256",
            ],
            "properties": {
                "model_version": {"type": "string"},
                "old_epoch": {"type": "integer"},
                "new_epoch": {"type": "integer"},
                "round_digest": {"type": "string"},
                "plan_checksum": {"type": "string"},
                "archive_sha256": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_rebuild_outside_refresh_hull",
        "description": "核验已完成刷新中明确转出边界的旧帧并重建相图；保留历史，不计算不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["expected_model", "expected_previous_version", "removed_record_ids"],
            "properties": {
                "expected_model": {"type": "string"},
                "expected_previous_version": {"type": "string"},
                "removed_record_ids": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_adopt_epoch_recovery_policy",
        "description": "仅按用户明确指令将已核验的旧恢复周期接入现行50标签规则，不训练、不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["expected_round_digest"],
            "properties": {"expected_round_digest": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_repair_refresh_handoff",
        "description": "按明确授权重打包已登记且未提交的刷新任务，保留旧包并添加远端相识别；不上传、不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["source_sha256", "task_count"],
            "properties": {"source_sha256": {"type": "string"}, "task_count": {"type": "integer"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_prepare_cpu_recovery",
        "description": "登记已提交刷新批次的CPU相识别恢复包；保留原任务与回收路径，不上传、不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["source_sha256", "spec_sha256", "task_count", "remote_root"],
            "properties": {
                "source_sha256": {"type": "string"},
                "spec_sha256": {"type": "string"},
                "task_count": {"type": "integer"},
                "remote_root": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_repair_dft_handoff",
        "description": "仅按用户明确授权修复已批准且未提交DFT包的单GPU MPI资源，保留任务及旧包，不上传不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["record_id", "source_sha256", "task_count"],
            "properties": {
                "record_id": {"type": "string"},
                "source_sha256": {"type": "string"},
                "task_count": {"type": "integer"},
                "partition_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_authorize_unknown_gs_dft",
        "description": "按用户明确指令，仅登记精确绑定、监测未知的当前基态独立DFT单点验证范围；不上传、不提交。",
        "inputSchema": {
            "type": "object",
            "required": [
                "candidate_frame_sha256",
                "expected_model",
                "expected_config",
                "expected_hull",
                "user_instruction",
            ],
            "properties": {
                "candidate_frame_sha256": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "expected_model": {"type": "string"},
                "expected_config": {"type": "string"},
                "expected_hull": {"type": "string"},
                "user_instruction": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_authorize_exact_candidate_relax",
        "description": "按用户明确指令登记冻结候选帧的弛豫输入准备授权；不生成、不上传、不提交。",
        "inputSchema": {
            "type": "object",
            "required": [
                "candidate_frame_sha256",
                "expected_model",
                "expected_config",
                "expected_hull",
                "user_instruction",
            ],
            "properties": {
                "candidate_frame_sha256": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "expected_model": {"type": "string"},
                "expected_config": {"type": "string"},
                "expected_hull": {"type": "string"},
                "user_instruction": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_register_endpoint_supplement",
        "description": "核验并登记当前刷新批次的历史末态外推证据；不计算、不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["csv_path", "archive_path", "csv_sha256", "archive_sha256"],
            "properties": {
                "csv_path": {"type": "string"},
                "archive_path": {"type": "string"},
                "csv_sha256": {"type": "string"},
                "archive_sha256": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_remove_quality_records",
        "description": "按用户明确授权，从当前凸包排除精确绑定的MLIP记录，保留原文件和审计，不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["expected_hull", "targets", "user_message"],
            "properties": {
                "expected_hull": {"type": "string"},
                "targets": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["geometry_key", "record_ids"],
                        "properties": {
                            "geometry_key": {"type": "string"},
                            "record_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "additionalProperties": False,
                    },
                },
                "user_message": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_restore_relocation_metadata",
        "description": "核验项目自己的旧控制备份，仅恢复当前状态仍引用的丢失路径迁移元数据。",
        "inputSchema": {
            "type": "object",
            "required": ["backup_path", "backup_sha256", "expected_revision"],
            "properties": {
                "backup_path": {"type": "string"},
                "backup_sha256": {"type": "string"},
                "expected_revision": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_propose_reopened_refresh",
        "description": "将已重开的冻结刷新送入原审批流程，生成待审记录；不会准备输入或提交。",
        "inputSchema": {
            "type": "object",
            "required": ["expected_preview"],
            "properties": {"expected_preview": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_refresh_dft_candidates",
        "description": "通过项目Agent刷新当前MLIP相图的DFT候选帧；不选择、准备输入或执行任务。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_settle_cancelled_dft_reservations",
        "description": "依据已登记的调度终态证据结算指定已取消 DFT 任务；不修改任务、不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["task_ids"],
            "properties": {"task_ids": {"type": "array", "items": {"type": "string"}}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_register_dft_wait_waiver",
        "description": "按用户明确指令记录指定 DFT 任务不再等待；仅结束本地跟踪，不向远端发送取消。",
        "inputSchema": {
            "type": "object",
            "required": ["task_ids", "user_message"],
            "properties": {
                "task_ids": {"type": "array", "items": {"type": "string"}},
                "user_message": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_register_scheduler_terminal_evidence",
        "description": "登记经外部调度器核对的终态证据；不修改任务结果、不结算费用、不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["observations"],
            "properties": {"observations": {"type": "array", "items": {"type": "object"}}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_register_prepared_unsubmitted_evidence",
        "description": "Record scoped SSH observations for prepared inputs; do not modify tasks or submit jobs.",
        "inputSchema": {
            "type": "object",
            "required": ["observation"],
            "properties": {"observation": {"type": "object"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_register_same_frame_comparisons",
        "description": "核验并登记已完成的同帧模型预测；不训练、不提交、不重算DFT。",
        "inputSchema": {
            "type": "object",
            "required": ["input_archive", "result_archive"],
            "properties": {
                "input_archive": {"type": "string"},
                "result_archive": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_register_diagnostic_retry",
        "description": "核验并登记已完成的独立DFT重试，不提交、不重算。",
        "inputSchema": {
            "type": "object",
            "required": ["input_archive", "result_archive"],
            "properties": {
                "input_archive": {"type": "string"},
                "result_archive": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_dft_recalculation_review",
        "description": "只读核对指定失败 DFT 任务并提供 Simulation 补算分析请求，不批准或执行。",
        "inputSchema": {
            "type": "object",
            "required": ["target_ids"],
            "properties": {
                "target_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 1000,
                    "uniqueItems": True,
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_register_dft_failure_analysis",
        "description": "登记原生 Agent 对已核验 DFT 大量失败的结构化诊断；不创建或提交任务。",
        "inputSchema": {
            "type": "object",
            "required": ["failure_evidence_digest", "diagnosis"],
            "properties": {
                "failure_evidence_digest": {"type": "string"},
                "diagnosis": {"type": "object"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_perturbation_recovery_candidates",
        "description": "只读核验外推正常、小扩胞的有效相图母结构；不选点、不生成任务。",
        "inputSchema": {
            "type": "object",
            "required": ["max_det_H"],
            "properties": {"max_det_H": {"type": "integer", "minimum": 1, "maximum": 6}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_direct_review_report",
        "description": "核对已登记 DIRECT 帧与母分支，生成供审核的派生报告；不执行或提交计算。",
        "inputSchema": {
            "type": "object",
            "required": ["scope_id"],
            "properties": {"scope_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_quarantine_extrapolation_refresh",
        "description": "按用户明确授权作废已完成 DIRECT 的刷新相图，恢复并验证上一版 latest，保留历史和任务。",
        "inputSchema": {
            "type": "object",
            "required": ["scope_id"],
            "properties": {"scope_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_bind_direct_candidates",
        "description": "核对已登记的 DIRECT 选中帧，补齐其母分支和结构信息；不创建 DFT 任务。",
        "inputSchema": {
            "type": "object",
            "required": ["scope_id"],
            "properties": {"scope_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_reopen_completed_refresh",
        "description": "按用户明确指令重开已完成刷新，保留旧数据与回执；不准备输入或提交。",
        "inputSchema": {
            "type": "object",
            "required": ["expected_model", "expected_plan", "expected_hull", "expected_preview"],
            "properties": {
                "expected_model": {"type": "string"},
                "expected_plan": {"type": "string"},
                "expected_hull": {"type": "string"},
                "expected_preview": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_preview_completed_refresh",
        "description": "只读预览已完成刷新在现行规则下的重算范围，不重开状态、不准备输入、不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["expected_model", "expected_plan", "expected_hull"],
            "properties": {
                "expected_model": {"type": "string"},
                "expected_plan": {"type": "string"},
                "expected_hull": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_archive_failed_epoch",
        "description": "按用户指令完整归档已作废的 epoch；验证文件哈希，保留旧路径和历史，不计算不提交。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_context",
        "description": "读取当前项目的 Codex 决策上下文、合法 action 和审批要求。默认返回紧凑摘要，传 include_details=true 获取完整配置与目录。",
        "inputSchema": {
            "type": "object",
            "properties": {"include_details": {"type": "boolean"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_pending",
        "description": "读取待审批方案。刷新可传 queue_revision；生成 branch 的明细可用 include_details=true 查看。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "if_queue_revision": {"type": "string"},
                "include_details": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_memory",
        "description": "读取 Agent 已保存的长短期记忆、经验候选和记忆审批队列；只读。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_propose",
        "description": "提交一个结构化 Codex action 到原有验证和审批流程。",
        "inputSchema": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {"type": "object"},
                "conversation_id": {"type": "string"},
                "include_details": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_decide",
        "description": "对已有待审批方案作出决定；批准以防重复后台操作执行，使用 phase_operation_status 查询结果。",
        "inputSchema": {
            "type": "object",
            "required": ["decision", "plan_id", "state_version", "proposal_hash"],
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["approve", "reject", "confirm_sensitive", "modify"],
                },
                "plan_id": {"type": "string"},
                "state_version": {"type": "string"},
                "proposal_hash": {"type": "string"},
                "comment": {"type": "string"},
                "include_details": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_operation_status",
        "description": "查询已登记后台操作的状态；不重放执行。",
        "inputSchema": {
            "type": "object",
            "required": ["operation_id"],
            "properties": {
                "operation_id": {"type": "string"},
                "include_details": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_report",
        "description": "为已有待审批方案生成可读报告；不会执行。",
        "inputSchema": {
            "type": "object",
            "required": ["plan_id"],
            "properties": {"plan_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_plot",
        "description": "使用已有数据绘制 MLIP；refresh_analysis 刷新分析归档，organize_analysis 整理目录和核验已登记选点，不提交计算。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "refresh_analysis": {"type": "boolean"},
                "training_parity": {"type": "boolean"},
                "round_dft_parity": {"type": "boolean"},
                "round_error_trend": {"type": "boolean"},
                "organize_analysis": {"type": "boolean"},
                "organize_dft": {
                    "type": "boolean",
                    "description": "Merge only the per-epoch DFT summary directories.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_list_projects",
        "description": "列出配置根目录下可供 Codex 接管的项目；只读，不修改项目。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_select_project",
        "description": "选择一个 agent_runtime.json 作为当前对话的项目；不会执行科学动作。",
        "inputSchema": {
            "type": "object",
            "required": ["runtime_config"],
            "properties": {"runtime_config": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_current_project",
        "description": "读取当前对话已选择的项目；只读。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_create_project",
        "description": "在指定路径创建新的隔离项目配置并自动选中；只创建配置，不启动计算或提交作业。",
        "inputSchema": {
            "type": "object",
            "required": ["workspace_root"],
            "properties": {
                "workspace_root": {"type": "string"},
                "execution_mode": {"type": "string", "enum": ["debug", "automatic"]},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_takeover_repo",
        "description": "确认当前 Codex MCP 已接管指定源码 repo；只读，不选择或修改科学项目。",
        "inputSchema": {
            "type": "object",
            "properties": {"repo_root": {"type": "string"}},
            "additionalProperties": False,
        },
    },
)


_TOOLS = (*_TOOLS, *TEAM_TOOLS)


_ANALYSIS_TOOLS = frozenset(
    {
        "phase_plot",
        "phase_publish_current_hull",
        "phase_list_projects",
        "phase_select_project",
        "phase_current_project",
        "phase_takeover_repo",
    }
)


_READ_ONLY_TOOLS = frozenset(
    {
        "phase_status",
        "phase_recovery_diagnostic",
        "phase_dft_recalculation_review",
        "phase_overview",
        "phase_pending",
        "phase_memory",
        "phase_list_projects",
        "phase_select_project",
        "phase_current_project",
    }
)


class CodexMcpServer:
    """JSON-RPC dispatcher used by the stdio MCP entry point and tests."""

    def __init__(
        self,
        codex: CodexOffline | None = None,
        *,
        runtime_config: str | os.PathLike[str] | None = None,
        project_roots: Iterable[str | os.PathLike[str]] = (),
        codex_factory: Callable[[str], CodexOffline] = CodexOffline,
        project_creator: Callable[[str, str], str | os.PathLike[str]] | None = None,
        repo_root: str | os.PathLike[str] | None = None,
        analysis_only: bool = False,
        read_only: bool = False,
    ):
        self.codex = codex
        self._needs_runtime = False
        self.analysis_only = analysis_only
        self.read_only = read_only
        if read_only and analysis_only:
            raise ValueError("read_only and analysis_only are separate access modes")
        if read_only:
            from phase_agent.runtime.saved_readonly_project import SavedReadOnlyProject

            codex_factory = SavedReadOnlyProject
        if analysis_only:
            from phase_agent.runtime.saved_analysis_project import SavedAnalysisProject

            codex_factory = SavedAnalysisProject
        self._codex_factory = codex_factory
        self._project_creator = project_creator or self._default_project_creator
        self.repo_root = Path(repo_root).expanduser().resolve() if repo_root else None
        self.runtime_config: Path | None = None
        self.project_roots = tuple(
            Path(root).expanduser().resolve() for root in project_roots if str(root).strip()
        )
        if runtime_config is not None:
            self._select_project(runtime_config)

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")
        if request_id is None and method not in {
            "notifications/initialized",
            "notifications/cancelled",
        }:
            return None
        if method == "initialize":
            return self._result(
                request_id,
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "phase-agent-codex", "version": "1.0"},
                },
            )
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return self._result(
                request_id,
                {
                    "tools": [
                        t
                        for t in _TOOLS
                        if (not self.analysis_only or t["name"] in _ANALYSIS_TOOLS)
                        and (not self.read_only or t["name"] in _READ_ONLY_TOOLS)
                    ]
                },
            )
        if method == "tools/call":
            try:
                params = message.get("params") or {}
                result = self._call_tool(
                    str(params.get("name") or ""), params.get("arguments") or {}
                )
                return self._result(
                    request_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, ensure_ascii=False, default=str),
                            }
                        ]
                    },
                )
            except LookupError as error:
                return self._error(request_id, -32601, str(error))
            except Exception as error:
                return self._error(request_id, -32000, str(error))
        return self._error(request_id, -32601, f"method not found: {method}")

    def _call_tool(self, name: str, args: dict[str, Any]) -> Any:
        if self.read_only and name not in _READ_ONLY_TOOLS:
            raise ValueError("read_only mode permits only saved status, memory and pending views")
        if self.analysis_only and name not in _ANALYSIS_TOOLS:
            raise ValueError("analysis_only mode does not permit workflow or configuration actions")
        if name == "phase_list_projects":
            return self._list_projects()
        if name == "phase_select_project":
            return self._select_project(args.get("runtime_config", ""))
        if name == "phase_current_project":
            return self._current_project()
        if name == "phase_create_project":
            return self._create_project(
                args.get("workspace_root", ""), args.get("execution_mode", "debug")
            )
        if name == "phase_takeover_repo":
            return self._takeover_repo(args.get("repo_root"))
        self._require_project(
            scientific_runtime=name
            not in {
                "phase_overview",
                "phase_status",
                "phase_pending",
                "phase_memory",
                "phase_dft_recalculation_review",
                "phase_repair_refresh_handoff",
                "phase_prepare_cpu_recovery",
                "phase_recovery_diagnostic",
            }
        )
        if name == "phase_dft_recalculation_review":
            if set(args) != {"target_ids"}:
                raise ValueError("invalid_dft_recalculation_review_arguments")
            return self.codex.control.dft_recalculation_review(args["target_ids"])
        if name in {"phase_recover", "phase_recovery_diagnostic"}:
            from phase_agent.runtime.recovery_diagnostics import (
                batch_recovery_diagnostic,
                compact_recovery_failure,
                failure_summary,
            )

            details = args.get("include_details", False)
            if type(details) is not bool:
                raise ValueError("include_details must be boolean")
            if name == "phase_recovery_diagnostic":
                batch_id = args.get("batch_id")
                if not isinstance(batch_id, str) or not 0 < len(batch_id) <= 1024:
                    raise ValueError("batch_id is required")
                return batch_recovery_diagnostic(
                    self.codex.control.state_path, batch_id, include_details=details
                )
            try:
                result = self.codex.recover()
            except Exception as error:
                if details:
                    raise
                return failure_summary("recovery", None, error, "inspect_saved_receipts")
            if details:
                return result
            return compact_recovery_failure(
                result, getattr(getattr(self.codex, "control", None), "state_path", None)
            )
        if name == "phase_recover_phase_evidence":
            return self.codex.recover_phase_evidence(**args)
        if name == "phase_publish_current_hull":
            if args:
                raise ValueError("phase_publish_current_hull_no_arguments")
            return self.codex.publish_phase()
        if name == "phase_overview":
            return self.codex.control.context_overview(publish=not self.read_only)
        if name == "phase_status":
            return self.codex.control.status()
        if name in TOOL_NAMES:
            return call_team_tool(self.codex.control, name, args)
        if name == "phase_set_remote_workspace":
            return self.codex.set_remote_workspace(args["remote_root"])
        if name == "phase_propose_recovery":
            return self.codex.control.propose_recovery(args)
        if name == "phase_hpc_recovery":
            return self.codex.control.hpc_startup_recovery()
        if name == "phase_review_relax_mc_return":
            quarantine = args.get("quarantine", False)
            if not isinstance(quarantine, bool):
                raise ValueError("quarantine must be boolean")
            return self.codex.control.review_relax_mc_return(
                args["bundle_id"], quarantine=quarantine
            )
        if name == "phase_minimize_unsent_relax_mc_delivery":
            return self.codex.minimize_unsent_relax_mc_delivery(**args)
        if name == "phase_register_relax_mc_input_attestation":
            return self.codex.register_relax_mc_input_attestation(**args)
        if name == "phase_preview_relax_mc_threshold_continuation":
            return self.codex.preview_relax_mc_threshold_continuation(args["bundle_id"])
        if name == "phase_register_relax_mc_threshold_transition":
            return self.codex.register_relax_mc_threshold_transition(**args)
        if name == "phase_name_direct_dft_handoff":
            if set(args) != {"record_id", "source_sha256", "remote_root"}:
                raise ValueError("invalid_direct_dft_naming_arguments")
            return self.codex.control.name_direct_dft_handoff(**args)
        if name == "phase_migrate_project_dft_nelm":
            if set(args) != {"expected_old", "expected_new", "value"}:
                raise ValueError("invalid_nelm_migration_arguments")
            return self.codex.control.migrate_project_dft_nelm(**args)
        if name == "phase_confirm_project_config":
            if set(args) != {"explicit"} or args["explicit"] is not True:
                raise ValueError("explicit_config_confirmation_required")
            return self.codex.confirm_config(explicit=True)
        if name == "phase_set_project_dft_nelm":
            if set(args) != {"value"}:
                raise ValueError("invalid_dft_nelm_arguments")
            return self.codex.control.set_project_dft_nelm(args["value"])
        if name == "phase_reconcile_recovery_refresh_epoch":
            if set(args) != {
                "model_version",
                "old_epoch",
                "new_epoch",
                "round_digest",
                "plan_checksum",
                "archive_sha256",
            }:
                raise ValueError("invalid_epoch_registration_arguments")
            return self.codex.control.reconcile_recovery_refresh_epoch(**args)
        if name == "phase_rebuild_outside_refresh_hull":
            if set(args) != {"expected_model", "expected_previous_version", "removed_record_ids"}:
                raise ValueError("invalid_outside_hull_rebuild_arguments")
            return self.codex.control.rebuild_mlip_lineage(
                expected_model=args["expected_model"],
                expected_previous_version=args["expected_previous_version"],
                required_record_ids=args["removed_record_ids"],
                reason="remove superseded outside-boundary frames",
            )
        if name == "phase_adopt_epoch_recovery_policy":
            if set(args) != {"expected_round_digest"}:
                raise ValueError("invalid_recovery_adoption_arguments")
            return self.codex.control.adopt_epoch_recovery_policy(args["expected_round_digest"])
        if name == "phase_repair_refresh_handoff":
            if set(args) != {"source_sha256", "task_count"}:
                raise ValueError("invalid_refresh_repair_arguments")
            if self._needs_runtime:
                return self.codex.control.repair_refresh_handoff(
                    **args, runtime_config_path=str(self.runtime_config)
                )
            return self.codex.control.repair_refresh_handoff(**args)
        if name == "phase_prepare_cpu_recovery":
            if set(args) != {"source_sha256", "spec_sha256", "task_count", "remote_root"}:
                raise ValueError("invalid_phase_cpu_recovery_arguments")
            return self.codex.control.prepare_phase_cpu_recovery(**args)
        if name == "phase_repair_dft_handoff":
            if not {"record_id", "source_sha256", "task_count"} <= set(args) or set(args) - {
                "record_id",
                "source_sha256",
                "task_count",
                "partition_counts",
            }:
                raise ValueError("invalid_dft_repair_arguments")
            return self.codex.control.repair_dft_handoff(
                args["record_id"],
                args["source_sha256"],
                args["task_count"],
                partition_counts=args.get("partition_counts"),
            )
        if name == "phase_authorize_unknown_gs_dft":
            required = {
                "candidate_frame_sha256",
                "expected_model",
                "expected_config",
                "expected_hull",
                "user_instruction",
            }
            if set(args) != required:
                raise ValueError("invalid_unknown_gs_dft_authorization_arguments")
            return self.codex.control.authorize_unknown_gs_dft(**args)
        if name == "phase_authorize_exact_candidate_relax":
            return self.codex.control.authorize_exact_candidate_relax(**args)
        if name == "phase_register_endpoint_supplement":
            if set(args) != {"csv_path", "archive_path", "csv_sha256", "archive_sha256"}:
                raise ValueError("invalid_endpoint_registration_arguments")
            return self.codex.control.register_endpoint_supplement(**args)
        if name == "phase_remove_quality_records":
            return self.codex.control.remove_phase_quality_records(**args)
        if name == "phase_propose_reopened_refresh":
            return self.codex.control.propose_reopened_refresh(**args)
        if name == "phase_restore_relocation_metadata":
            from phase_agent.runtime.restore_relocation_metadata import restore_relocation_metadata

            return restore_relocation_metadata(self.codex.control.state_path, **args)
        if name == "phase_refresh_dft_candidates":
            return self.codex.control.refresh_dft_candidates()
        if name == "phase_settle_cancelled_dft_reservations":
            return self.codex.control.settle_cancelled_dft_reservations(**args)
        if name == "phase_register_dft_wait_waiver":
            return self.codex.control.register_dft_wait_waiver(**args)
        if name == "phase_register_scheduler_terminal_evidence":
            if set(args) != {"observations"}:
                raise ValueError("invalid_scheduler_evidence_arguments")
            return self.codex.control.register_scheduler_terminal_evidence(args["observations"])
        if name == "phase_register_prepared_unsubmitted_evidence":
            if set(args) != {"observation"}:
                raise ValueError("invalid_prepared_unsubmitted_evidence_arguments")
            return self.codex.control.register_prepared_unsubmitted_evidence(args["observation"])
        if name == "phase_register_same_frame_comparisons":
            if set(args) != {"input_archive", "result_archive"}:
                raise ValueError("invalid_same_frame_comparison_arguments")
            return self.codex.control.register_same_frame_comparisons(**args)
        if name == "phase_register_diagnostic_retry":
            if set(args) != {"input_archive", "result_archive"}:
                raise ValueError("invalid_diagnostic_retry_arguments")
            return self.codex.control.register_diagnostic_retry(**args)
        if name == "phase_register_dft_failure_analysis":
            if set(args) != {"failure_evidence_digest", "diagnosis"}:
                raise ValueError("invalid_dft_failure_analysis_arguments")
            return self.codex.control.register_dft_failure_analysis(**args)
        if name == "phase_perturbation_recovery_candidates":
            return self.codex.control.perturbation_recovery_candidates(args["max_det_H"])
        if name == "phase_direct_review_report":
            return self.codex.control.direct_review_report(args["scope_id"])
        if name == "phase_quarantine_extrapolation_refresh":
            return self.codex.control.quarantine_extrapolation_refresh(args["scope_id"])
        if name == "phase_bind_direct_candidates":
            return self.codex.control.bind_direct_candidates(args["scope_id"])
        if name == "phase_reopen_completed_refresh":
            return self.codex.reopen_completed_refresh(**args)
        if name == "phase_preview_completed_refresh":
            return self.codex.preview_completed_refresh(**args)
        if name == "phase_archive_failed_epoch":
            return self.codex.control.archive_failed_epoch()
        if name == "phase_regenerate_training_preflight":
            expected = {
                "task_id",
                "input_structure_sha256",
                "model_version",
                "expected_job_key",
                "independent_test_fraction",
                "seed",
                "user_instruction",
            }
            if set(args) != expected:
                raise ValueError("invalid_training_preflight_arguments")
            return self.codex.control.regenerate_training_preflight(**args)
        if name == "phase_context":
            include_details = args.get("include_details", False)
            if type(include_details) is not bool:
                raise ValueError("include_details must be boolean")
            ctx = self.codex.context()
            if include_details or not isinstance(ctx, dict):
                return ctx
            return compact_context_response(ctx)
        if name == "phase_pending":
            known = args.get("if_queue_revision")
            if known is not None and not isinstance(known, str):
                raise ValueError("if_queue_revision must be a string")
            details = args.get("include_details", False)
            if type(details) is not bool:
                raise ValueError("include_details must be boolean")
            pending = self.codex.pending()
            revision = sha256(
                json.dumps(
                    [details, pending], ensure_ascii=False, sort_keys=True, default=str
                ).encode()
            ).hexdigest()
            if known == revision:
                return {
                    "status": "unchanged",
                    "queue_revision": revision,
                    "state_version": pending["state_version"],
                    "count": pending["count"],
                }
            return {
                **(pending if details else compact_pending_response(pending)),
                "queue_revision": revision,
            }
        if name == "phase_memory":
            return self.codex.control.memory()
        if name == "phase_propose":
            include_details = args.get("include_details", False)
            if type(include_details) is not bool:
                raise ValueError("include_details must be boolean")
            res = self.codex.propose(
                args["action"], conversation_id=args.get("conversation_id", "codex-mcp")
            )
            if include_details or not isinstance(res, dict):
                return res
            return {k: v for k, v in res.items() if k not in _HEAVY_MCP_FIELDS}
        if name == "phase_decide":
            include_details = args.get("include_details", False)
            if type(include_details) is not bool:
                raise ValueError("include_details must be boolean")
            codex = self.codex
            operations = getattr(getattr(codex, "control", None), "operations", None)
            if args["decision"] in {"approve", "confirm_sensitive"} and operations is not None:
                request = {
                    "route": "phase_decide",
                    "decision": args["decision"],
                    "plan_id": args["plan_id"],
                    "state_version": args["state_version"],
                    "proposal_hash": args["proposal_hash"],
                    "comment": args.get("comment", ""),
                    "include_details": include_details,
                }
                request_id = "approval:" + args["plan_id"] + ":" + args["proposal_hash"]

                def activate():
                    result = codex.decide(
                        request["decision"],
                        plan_id=request["plan_id"],
                        state_version=request["state_version"],
                        proposal_hash=request["proposal_hash"],
                        comment=request["comment"],
                    )
                    return (
                        result
                        if request["include_details"]
                        else compact_decision_response(result, request["plan_id"])
                    )

                return operations.submit(activate, request_id=request_id, request=request)
            result = self.codex.decide(
                args["decision"],
                plan_id=args["plan_id"],
                state_version=args["state_version"],
                proposal_hash=args["proposal_hash"],
                comment=args.get("comment", ""),
            )
            return result if include_details else compact_decision_response(result, args["plan_id"])
        if name == "phase_operation_status":
            include_details = args.get("include_details", False)
            if type(include_details) is not bool:
                raise ValueError("include_details must be boolean")
            operations = getattr(getattr(self.codex, "control", None), "operations", None)
            if operations is None:
                raise ValueError("background_control_unavailable")
            op = operations.get(args["operation_id"])
            if include_details or not isinstance(op, dict):
                return op
            return compact_operation_status_response(op)
        if name == "phase_report":
            return self.codex.control.codex_report(args["plan_id"])
        if name == "phase_plot":
            organize_dft = args.get("organize_dft", False)
            if not isinstance(organize_dft, bool):
                raise ValueError("organize_dft must be boolean")
            if organize_dft:
                if any(value for key, value in args.items() if key != "organize_dft"):
                    raise ValueError("organize_dft must be requested separately")
                return self.codex.control.codex_plot(organize_dft=True)
            refresh = args.get("refresh_analysis", False)
            if not isinstance(refresh, bool):
                raise ValueError("refresh_analysis must be boolean")
            organize = args.get("organize_analysis", False)
            if not isinstance(organize, bool):
                raise ValueError("organize_analysis must be boolean")
            training = args.get("training_parity", False)
            if not isinstance(training, bool):
                raise ValueError("training_parity must be boolean")
            trend = args.get("round_error_trend", False)
            if not isinstance(trend, bool):
                raise ValueError("round_error_trend must be boolean")
            if trend:
                return self.codex.control.codex_plot(round_error_trend=True)
            round_dft = args.get("round_dft_parity", False)
            if not isinstance(round_dft, bool):
                raise ValueError("round_dft_parity must be boolean")
            if round_dft:
                return self.codex.control.codex_plot(
                    refresh_analysis=refresh,
                    organize_analysis=organize,
                    training_parity=training,
                    round_dft_parity=True,
                )
            if training:
                return self.codex.control.codex_plot(
                    refresh_analysis=refresh, organize_analysis=organize, training_parity=True
                )
            if organize:
                return self.codex.control.codex_plot(
                    refresh_analysis=refresh, organize_analysis=True
                )
            if refresh:
                return self.codex.control.codex_plot(refresh_analysis=True)
            return self.codex.control.codex_plot()
        raise LookupError(f"tool not found: {name}")

    def _require_project(self, *, scientific_runtime=True) -> None:
        if self.codex is None:
            raise RuntimeError("尚未选择项目；请先调用 phase_list_projects 或 phase_select_project")
        if scientific_runtime and self._needs_runtime:
            # Construct once only when an operation needs the normal workflow.
            # Failure retains the read-only binding and does not retry an action.
            runtime = self._codex_factory(str(self.runtime_config))
            self.codex = runtime
            self._needs_runtime = False

    def _select_project(self, runtime_config: str | os.PathLike[str]) -> dict[str, Any]:
        value = str(runtime_config or "").strip()
        if not value:
            raise ValueError("runtime_config 不能为空")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"项目运行配置不存在：{path}")
        if path.name.lower() != "agent_runtime.json":
            raise ValueError("只能选择名为 agent_runtime.json 的项目配置")
        needs_runtime = False
        if self._codex_factory is CodexOffline:
            from phase_agent.runtime.saved_readonly_project import SavedReadOnlyProject

            try:
                codex = SavedReadOnlyProject(str(path))
                needs_runtime = True
            except FileNotFoundError:
                # New unsaved/draft projects retain their existing initialization.
                codex = self._codex_factory(str(path))
        else:
            codex = self._codex_factory(str(path))
        self.codex = codex
        self._needs_runtime = needs_runtime
        self.runtime_config = path
        return self._current_project()

    def _current_project(self) -> dict[str, Any]:
        if self.runtime_config is None:
            return {"selected": False, "runtime_config": None, "name": None}
        return {
            "selected": True,
            "runtime_config": str(self.runtime_config),
            "name": self.runtime_config.parent.name,
        }

    @staticmethod
    def _default_project_creator(workspace_root: str, execution_mode: str) -> str:
        from phase_agent.runtime.local_project_launcher import create_project

        return str(create_project(Path(workspace_root), execution_mode=execution_mode))

    def _create_project(self, workspace_root: str, execution_mode: str) -> dict[str, Any]:
        value = str(workspace_root or "").strip()
        if not value:
            raise ValueError("workspace_root 不能为空")
        if execution_mode not in {"debug", "automatic"}:
            raise ValueError("execution_mode 必须是 debug 或 automatic")
        runtime_config = self._project_creator(value, execution_mode)
        selected = self._select_project(runtime_config)
        return {"created": True, "execution_mode": execution_mode, **selected}

    def _takeover_repo(self, requested_repo: str | None) -> dict[str, Any]:
        path = Path(requested_repo).expanduser().resolve() if requested_repo else self.repo_root
        if path is None:
            raise ValueError("MCP 未绑定 repo；请在启动参数中传入 --repo-root")
        if not path.is_dir():
            raise FileNotFoundError(f"repo 不存在：{path}")
        required = (path / "phase_agent", path / "codex-agent.cmd")
        missing = [str(item) for item in required if not item.exists()]
        if missing:
            raise ValueError(f"不是可接管的 phase_agent repo，缺少：{', '.join(missing)}")
        from copy import deepcopy
        from phase_agent.runtime.codex.context import HPC_APPROVAL_POLICY

        self.repo_root = path
        from phase_agent.runtime.codex.team import native_team_workflow

        return {
            "codex_team_workflow": native_team_workflow(),
            "hpc_approval_policy": deepcopy(HPC_APPROVAL_POLICY),
            "status": "ready",
            "repo_root": str(path),
            "entrypoint": str(path / "codex-mcp.cmd"),
            "next": ["phase_list_projects", "phase_select_project", "phase_create_project"],
        }

    def _list_projects(self) -> dict[str, Any]:
        candidates: set[Path] = set()
        for root in self.project_roots:
            if root.is_file() and root.name.lower() == "agent_runtime.json":
                candidates.add(root)
                continue
            if not root.is_dir():
                continue
            direct = root / "agent_runtime.json"
            if direct.is_file():
                candidates.add(direct)
            try:
                children = tuple(root.iterdir())
            except OSError:
                continue
            for child in children:
                if child.is_dir():
                    candidate = child / "agent_runtime.json"
                    if candidate.is_file():
                        candidates.add(candidate)
        projects = []
        for path in sorted(candidates, key=lambda item: str(item).lower()):
            projects.append(
                {
                    "name": path.parent.name,
                    "runtime_config": str(path.resolve()),
                    "selected": path.resolve() == self.runtime_config,
                }
            )
        return {"projects": projects, "roots": [str(root) for root in self.project_roots]}

    @staticmethod
    def _result(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


_HEAVY_MCP_FIELDS = {
    "effective_config",
    "scientific_operations",
    "calculation_cost_reference",
    "state",
    "team_state",
}


def compact_context_response(ctx):
    """Strip static catalogs and full config/state from phase_context by default."""
    clean = {k: v for k, v in ctx.items() if k not in _HEAVY_MCP_FIELDS}
    omitted = sorted(k for k in ctx if k in _HEAVY_MCP_FIELDS)
    if omitted:
        clean["details_omitted"] = omitted
    return clean


def compact_pending_response(response):
    """Keep actionable sampling parameters and summarize detailed preview evidence."""
    rows = []
    for row in response["pending"]:
        if not isinstance(row, dict):
            rows.append(row)
            continue
        clean_row = {k: v for k, v in row.items() if k not in _HEAVY_MCP_FIELDS}
        omitted = [k for k in row if k in _HEAVY_MCP_FIELDS]
        if isinstance(clean_row.get("target_ids"), list) and len(clean_row["target_ids"]) > 5:
            clean_row["target_count"] = len(clean_row["target_ids"])
            clean_row["target_ids_preview"] = clean_row.pop("target_ids")[:5]
            omitted.append("target_ids")
        params = clean_row.get("parameters")
        if isinstance(params, dict):
            params = {k: v for k, v in params.items() if k not in _HEAVY_MCP_FIELDS}
            if isinstance(params.get("target_ids"), list) and len(params["target_ids"]) > 5:
                params["target_count"] = len(params["target_ids"])
                params["target_ids_preview"] = params.pop("target_ids")[:5]
                omitted.append("parameters.target_ids")
            clean_row["parameters"] = params
        preview = params.get("region_preview") if isinstance(params, dict) else None
        if clean_row.get("recommended_action") == "generate_branches" and isinstance(preview, dict):
            preview = dict(preview)
            evidence = preview.get("near_hull_parent_evidence")
            if isinstance(evidence, dict) and isinstance(evidence.get("parents"), (list, dict)):
                evidence = dict(evidence)
                evidence["parent_count"] = len(evidence.pop("parents"))
                preview["near_hull_parent_evidence"] = evidence
                omitted.append("region_preview.near_hull_parent_evidence.parents")
            if isinstance(preview.get("regions"), list):
                regions = []
                for region in preview["regions"]:
                    if isinstance(region, dict) and isinstance(region.get("cells"), list):
                        region = {**region, "cell_count": len(region["cells"])}
                        del region["cells"]
                        omitted.append("region_preview.regions.cells")
                    regions.append(region)
                preview["regions"] = regions
            clean_row["parameters"] = {**params, "region_preview": preview}
        if omitted:
            existing = clean_row.get("details_omitted") or []
            clean_row["details_omitted"] = sorted(set([*existing, *omitted]))
        rows.append(clean_row)
    first = rows[0] if rows and isinstance(rows[0], dict) else {}
    base = {k: v for k, v in response.items() if k not in _HEAVY_MCP_FIELDS}
    return {
        **base,
        "plan_id": base.get("plan_id") or first.get("plan_id"),
        "proposal_hash": base.get("proposal_hash") or first.get("proposal_hash"),
        "pending": rows,
    }


def compact_operation_status_response(op):
    """Strip heavy config/state catalogs from phase_operation_status by default."""
    clean = {k: v for k, v in op.items() if k not in _HEAVY_MCP_FIELDS}
    result = clean.get("result")
    plan_id = clean.get("plan_id")
    proposal_hash = clean.get("proposal_hash")
    state_version = clean.get("state_version")
    if isinstance(result, dict):
        plan_id = plan_id or result.get("plan_id")
        proposal_hash = proposal_hash or result.get("proposal_hash")
        state_version = state_version or result.get("state_version")
        compacted = compact_decision_response(result, plan_id or "")
        if isinstance(compacted, dict):
            compacted = {k: v for k, v in compacted.items() if k not in _HEAVY_MCP_FIELDS}
        clean["result"] = compacted
    req = clean.get("request")
    if isinstance(req, dict):
        plan_id = plan_id or req.get("plan_id")
        proposal_hash = proposal_hash or req.get("proposal_hash")
        state_version = state_version or req.get("state_version")
    if plan_id is not None:
        clean["plan_id"] = plan_id
    if proposal_hash is not None:
        clean["proposal_hash"] = proposal_hash
    if state_version is not None:
        clean["state_version"] = state_version
    return clean


def compact_decision_response(response, plan_id):
    """Keep execution status and handoff references out of the project-state payload."""
    hidden = {
        "state",
        "event_state",
        "loaded_state",
        "decision_context",
        "agent_state",
        "relax_mc_preview",
        "refresh_preview",
        "energy_repair_preview",
        "budget_preview",
        "raw_action",
        "final_action",
        "agent_proposal",
    }
    omitted = ["state"] if "state" in response else []

    def project(value):
        if isinstance(value, dict):
            for key in value.keys() & hidden:
                omitted.append(key)
            result = {}
            branch_execution = value.get("tool") == "generate_branches"
            for key, item in value.items():
                if key in hidden:
                    continue
                if (
                    key == "action"
                    and isinstance(item, dict)
                    and isinstance(value.get("execution") or value.get("execution_result"), dict)
                ):
                    omitted.append(key)
                    continue
                if key == "result" and branch_execution and isinstance(item, dict):
                    item = dict(item)
                    for field, count_key in (
                        ("registered", "registered_count"),
                        ("coverage", "coverage_row_count"),
                    ):
                        if isinstance(item.get(field), list):
                            rows = item.pop(field)
                            item.setdefault(count_key, len(rows))
                            omitted.append(field)
                result[key] = project(item)
            if (
                "execution" in result
                and "execution_result" in result
                and value["execution"] == value["execution_result"]
            ):
                omitted.append("execution_result")
                del result["execution_result"]
            return result
        if isinstance(value, list):
            return [project(item) for item in value]
        return value

    result = project(response)
    result["plan_id"] = plan_id
    if omitted:
        result["details_omitted"] = sorted(set(omitted))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过 stdio MCP 暴露项目 Codex 接管入口")
    parser.add_argument("--runtime-config")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--analysis-only",
        action="store_true",
        help="只读加载已有数据；仅开放分析绘图与归档，不初始化或迁移科研运行时",
    )
    modes.add_argument(
        "--read-only",
        action="store_true",
        help="只读取已保存状态、记忆和待审方案，不初始化/迁移运行时或生成文件",
    )
    parser.add_argument(
        "--project-root",
        action="append",
        default=[],
        help="可发现项目的根目录；可重复传入多个目录",
    )
    parser.add_argument("--repo-root", help="Codex 接管的 phase_agent 源码 repo 根目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Windows consoles commonly default to GBK; MCP payloads contain Chinese
    # project evidence and must always use UTF-8 JSON in both directions.
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    project_roots = list(args.project_root)
    if not project_roots:
        project_roots = [
            value for value in os.environ.get("PHASE_PROJECT_ROOTS", "").split(os.pathsep) if value
        ]
    # Scientific libraries may print progress; stdout is reserved for JSON-RPC.
    from contextlib import redirect_stdout

    protocol_stdout = sys.stdout
    with redirect_stdout(sys.stderr):
        server = CodexMcpServer(
            runtime_config=args.runtime_config,
            project_roots=project_roots,
            repo_root=args.repo_root,
            analysis_only=args.analysis_only,
            read_only=args.read_only,
        )
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            with redirect_stdout(sys.stderr):
                response = server.handle(message)
            if response is not None:
                protocol_stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
                protocol_stdout.flush()
        except Exception as error:
            response = CodexMcpServer._error(None, -32700, str(error))
            protocol_stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            protocol_stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
