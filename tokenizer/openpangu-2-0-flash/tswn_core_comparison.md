# tswn_core Rust 源码 tokenizer 对比

## 统计口径

| 项目 | 值 |
|---|---|
| 输入目录 | `\\?\D:\githubs\namer\tswn-core\crates\tswn_core\src` |
| 文件数 | 116 |
| Tokenizer 数 | 3 |
| 主对比 tokenizer | OpenPangu |
| 编码方式 | `Tokenizer::encode(text, false)`，不添加 special tokens |
| OpenPangu tokenizer | `tokenizer/openpangu-2-0-flash/tokenizer.json` |
| V4P tokenizer | `tokenizer/deepseek-v4-pro/tokenizer.json` |
| GLM 5.2 tokenizer | `tokenizer/glm-5-2/tokenizer.json` |

## 总体结果

| Tokenizer | 总 token 数 | 平均每项 | 中位数 | 最小 | 最大 | 相对主 tokenizer 差值 | 相对主 tokenizer 增幅 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenPangu | 518,841 | 4472.77 | 2035 | 198 | 56,520 | +0 | +0.00% |
| V4P | 546,778 | 4713.60 | 2593 | 205 | 43,055 | +27937 | +5.38% |
| GLM 5.2 | 470,155 | 4053.06 | 1992 | 196 | 42,356 | -48686 | -9.38% |

## 按分组汇总

| 分组 | 项数 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|---:|
| . | 8 | 84,818 | 26,478 | 30,499 | 24,680 | 1.15186x | 0.93209x |
| bin | 6 | 177,975 | 46,158 | 54,184 | 45,317 | 1.17388x | 0.98178x |
| bin/tswn_cli | 3 | 16,647 | 4,714 | 5,770 | 4,631 | 1.22401x | 0.98239x |
| bin/tswn_cli/args | 4 | 46,735 | 13,932 | 17,430 | 13,885 | 1.25108x | 0.99663x |
| bin/tswn_cli/bench | 6 | 70,002 | 18,633 | 23,089 | 18,129 | 1.23915x | 0.97295x |
| bin/tswn_cli/fight | 4 | 34,419 | 9,700 | 9,985 | 9,496 | 1.02938x | 0.97897x |
| cli_api | 3 | 32,769 | 8,386 | 10,823 | 8,142 | 1.29060x | 0.97090x |
| engine | 10 | 119,703 | 35,752 | 42,273 | 35,412 | 1.18240x | 0.99049x |
| player | 15 | 356,137 | 135,449 | 128,865 | 116,354 | 0.95139x | 0.85902x |
| player/boss | 5 | 56,767 | 15,001 | 19,277 | 14,670 | 1.28505x | 0.97793x |
| player/icon_render | 2 | 101,635 | 95,436 | 74,873 | 74,034 | 0.78454x | 0.77575x |
| player/skill | 1 | 28,523 | 7,276 | 7,698 | 7,164 | 1.05800x | 0.98461x |
| player/skill/act | 28 | 160,403 | 43,894 | 53,674 | 42,515 | 1.22281x | 0.96858x |
| player/skill/skl | 13 | 64,020 | 16,331 | 19,814 | 15,886 | 1.21328x | 0.97275x |
| player/test | 8 | 153,082 | 41,701 | 48,524 | 39,840 | 1.16362x | 0.95537x |

## 逐项差异分位数

| Tokenizer | 分位数 | 相对主 tokenizer 差值 | 相对主 tokenizer 比例 |
|---|---|---:|---:|
| V4P | 最小 | -20410 | 0.63889x |
| V4P | P10 | +44 | 1.04369x |
| V4P | P25 | +174 | 1.16859x |
| V4P | P50 | +360 | 1.22306x |
| V4P | P75 | +740 | 1.26183x |
| V4P | P90 | +1502 | 1.29126x |
| V4P | P99 | +2814 | 1.34974x |
| V4P | 最大 | +2943 | 1.35859x |
| GLM 5.2 | 最小 | -16767 | 0.70334x |
| GLM 5.2 | P10 | -244 | 0.95719x |
| GLM 5.2 | P25 | -90 | 0.96412x |
| GLM 5.2 | P50 | -52 | 0.97305x |
| GLM 5.2 | P75 | -25 | 0.98584x |
| GLM 5.2 | P90 | -2 | 0.99692x |
| GLM 5.2 | P99 | +17 | 1.02228x |
| GLM 5.2 | 最大 | +33 | 1.02525x |

## 分 tokenizer 视角结论

### 以 OpenPangu 为核心

| 对比对象 | 核心总量 | 对象总量 | 对象 - 核心 | 对象 / 核心 | 逐项对象更少 | 逐项相同 | 逐项对象更多 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| V4P | 518,841 | 546,778 | +27937 | 1.05385x | 5 | 0 | 111 | V4P 比 OpenPangu 多 5.38%，OpenPangu 更节省 token |
| GLM 5.2 | 518,841 | 470,155 | -48686 | 0.90616x | 106 | 1 | 9 | GLM 5.2 比 OpenPangu 少 9.38%，OpenPangu 使用更多 token |

### 以 V4P 为核心

| 对比对象 | 核心总量 | 对象总量 | 对象 - 核心 | 对象 / 核心 | 逐项对象更少 | 逐项相同 | 逐项对象更多 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| OpenPangu | 546,778 | 518,841 | -27937 | 0.94891x | 111 | 0 | 5 | OpenPangu 比 V4P 少 5.11%，V4P 使用更多 token |
| GLM 5.2 | 546,778 | 470,155 | -76623 | 0.85986x | 115 | 0 | 1 | GLM 5.2 比 V4P 少 14.01%，V4P 使用更多 token |

### 以 GLM 5.2 为核心

| 对比对象 | 核心总量 | 对象总量 | 对象 - 核心 | 对象 / 核心 | 逐项对象更少 | 逐项相同 | 逐项对象更多 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| OpenPangu | 470,155 | 518,841 | +48686 | 1.10355x | 9 | 1 | 106 | OpenPangu 比 GLM 5.2 多 10.36%，GLM 5.2 更节省 token |
| V4P | 470,155 | 546,778 | +76623 | 1.16297x | 1 | 0 | 115 | V4P 比 GLM 5.2 多 16.30%，GLM 5.2 更节省 token |

## 逐项结果

### .

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `bench_sched.rs` | 7,398 | 2,032 | 2,564 | 1,985 | 1.26181x | 0.97687x |
| `case_gen.rs` | 6,084 | 1,701 | 1,715 | 1,640 | 1.00823x | 0.96414x |
| `debug.rs` | 6,599 | 1,868 | 2,532 | 1,857 | 1.35546x | 0.99411x |
| `error.rs` | 4,384 | 1,118 | 1,400 | 1,126 | 1.25224x | 1.00716x |
| `lib.rs` | 2,504 | 725 | 917 | 694 | 1.26483x | 0.95724x |
| `rc4.rs` | 24,753 | 10,364 | 10,340 | 8,889 | 0.99768x | 0.85768x |
| `replay_view.rs` | 27,077 | 6,969 | 9,001 | 6,815 | 1.29158x | 0.97790x |
| `win_rate.rs` | 6,019 | 1,701 | 2,030 | 1,674 | 1.19342x | 0.98413x |

### bin

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `bin/track.rs` | 1,818 | 538 | 680 | 538 | 1.26394x | 1.00000x |
| `bin/track_case_miner.rs` | 32,066 | 7,888 | 10,183 | 7,834 | 1.29095x | 0.99315x |
| `bin/track_diy_roundtrip.rs` | 40,319 | 10,382 | 13,101 | 10,215 | 1.26190x | 0.98391x |
| `bin/track_perf_cases.rs` | 28,787 | 7,817 | 8,181 | 7,568 | 1.04657x | 0.96815x |
| `bin/track_test.rs` | 23,681 | 5,838 | 7,566 | 5,760 | 1.29599x | 0.98664x |
| `bin/tswn_case_miner.rs` | 51,304 | 13,695 | 14,473 | 13,402 | 1.05681x | 0.97861x |

### bin/tswn_cli

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `bin/tswn_cli/icon.rs` | 3,512 | 1,094 | 1,150 | 1,061 | 1.05119x | 0.96984x |
| `bin/tswn_cli/main.rs` | 8,003 | 2,245 | 2,861 | 2,247 | 1.27439x | 1.00089x |
| `bin/tswn_cli/to_diy.rs` | 5,132 | 1,375 | 1,759 | 1,323 | 1.27927x | 0.96218x |

### bin/tswn_cli/args

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `bin/tswn_cli/args/cli.rs` | 32,135 | 8,916 | 11,324 | 8,890 | 1.27008x | 0.99708x |
| `bin/tswn_cli/args/input.rs` | 7,499 | 2,488 | 3,033 | 2,429 | 1.21905x | 0.97629x |
| `bin/tswn_cli/args/mod.rs` | 501 | 221 | 266 | 226 | 1.20362x | 1.02262x |
| `bin/tswn_cli/args/parsed.rs` | 6,600 | 2,307 | 2,807 | 2,340 | 1.21673x | 1.01430x |

### bin/tswn_cli/bench

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `bin/tswn_cli/bench/batch.rs` | 26,346 | 6,517 | 8,443 | 6,341 | 1.29553x | 0.97299x |
| `bin/tswn_cli/bench/common.rs` | 1,232 | 459 | 550 | 441 | 1.19826x | 0.96078x |
| `bin/tswn_cli/bench/mod.rs` | 559 | 214 | 230 | 215 | 1.07477x | 1.00467x |
| `bin/tswn_cli/bench/output.rs` | 9,608 | 2,748 | 2,927 | 2,711 | 1.06514x | 0.98654x |
| `bin/tswn_cli/bench/score.rs` | 21,331 | 5,808 | 7,259 | 5,628 | 1.24983x | 0.96901x |
| `bin/tswn_cli/bench/winrate.rs` | 10,926 | 2,887 | 3,680 | 2,793 | 1.27468x | 0.96744x |

### bin/tswn_cli/fight

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `bin/tswn_cli/fight/driver.rs` | 6,764 | 1,862 | 1,938 | 1,816 | 1.04082x | 0.97530x |
| `bin/tswn_cli/fight/mod.rs` | 451 | 198 | 205 | 203 | 1.03535x | 1.02525x |
| `bin/tswn_cli/fight/raw_bench.rs` | 11,819 | 3,445 | 3,551 | 3,356 | 1.03077x | 0.97417x |
| `bin/tswn_cli/fight/trace.rs` | 15,385 | 4,195 | 4,291 | 4,121 | 1.02288x | 0.98236x |

### cli_api

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `cli_api/bench.rs` | 10,805 | 2,738 | 3,497 | 2,646 | 1.27721x | 0.96640x |
| `cli_api/mod.rs` | 13,262 | 3,369 | 4,438 | 3,298 | 1.31730x | 0.97893x |
| `cli_api/parse.rs` | 8,702 | 2,279 | 2,888 | 2,198 | 1.26722x | 0.96446x |

### engine

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `engine/engine_core.rs` | 11,474 | 3,076 | 3,369 | 3,022 | 1.09525x | 0.98244x |
| `engine/hooks.rs` | 4,534 | 1,333 | 1,678 | 1,350 | 1.25881x | 1.01275x |
| `engine/lang.rs` | 20,982 | 7,574 | 8,828 | 7,540 | 1.16557x | 0.99551x |
| `engine/mod.rs` | 1,716 | 688 | 800 | 702 | 1.16279x | 1.02035x |
| `engine/rules.rs` | 653 | 224 | 291 | 227 | 1.29911x | 1.01339x |
| `engine/runners.rs` | 25,117 | 6,952 | 7,422 | 6,880 | 1.06761x | 0.98964x |
| `engine/storage.rs` | 22,448 | 6,422 | 7,974 | 6,304 | 1.24167x | 0.98163x |
| `engine/tick.rs` | 8,884 | 2,330 | 2,964 | 2,304 | 1.27210x | 0.98884x |
| `engine/update.rs` | 9,084 | 2,935 | 3,697 | 2,922 | 1.25963x | 0.99557x |
| `engine/world_state.rs` | 14,811 | 4,218 | 5,250 | 4,161 | 1.24467x | 0.98649x |

### player

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `player/action_targets.rs` | 1,747 | 518 | 677 | 488 | 1.30695x | 0.94208x |
| `player/eval_name.rs` | 67,224 | 56,520 | 36,110 | 39,753 | 0.63889x | 0.70334x |
| `player/icon.rs` | 10,426 | 3,877 | 4,490 | 3,639 | 1.15811x | 0.93861x |
| `player/icon_render.rs` | 6,236 | 2,366 | 2,773 | 2,281 | 1.17202x | 0.96407x |
| `player/impl_attr.rs` | 40,626 | 11,034 | 13,392 | 10,647 | 1.21370x | 0.96493x |
| `player/impl_ctor.rs` | 14,262 | 3,973 | 4,810 | 3,848 | 1.21067x | 0.96854x |
| `player/impl_runtime.rs` | 93,094 | 20,683 | 22,377 | 20,119 | 1.08190x | 0.97273x |
| `player/mod.rs` | 8,474 | 3,218 | 3,788 | 3,116 | 1.17713x | 0.96830x |
| `player/overlay.rs` | 14,374 | 4,381 | 5,411 | 4,318 | 1.23511x | 0.98562x |
| `player/skill.rs` | 47,045 | 13,880 | 16,711 | 13,491 | 1.20396x | 0.97197x |
| `player/state.rs` | 27,354 | 6,998 | 8,847 | 6,856 | 1.26422x | 0.97971x |
| `player/status.rs` | 4,957 | 1,644 | 2,000 | 1,622 | 1.21655x | 0.98662x |
| `player/test.rs` | 2,081 | 583 | 601 | 563 | 1.03087x | 0.96569x |
| `player/utils.rs` | 2,937 | 999 | 1,160 | 923 | 1.16116x | 0.92392x |
| `player/weapons.rs` | 15,300 | 4,775 | 5,718 | 4,690 | 1.19749x | 0.98220x |

### player/boss

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `player/boss/covid.rs` | 19,369 | 4,627 | 6,011 | 4,575 | 1.29911x | 0.98876x |
| `player/boss/lazy.rs` | 7,833 | 2,158 | 2,762 | 2,097 | 1.27989x | 0.97173x |
| `player/boss/mod.rs` | 9,293 | 2,535 | 3,224 | 2,458 | 1.27179x | 0.96963x |
| `player/boss/saitama.rs` | 3,874 | 1,038 | 1,296 | 1,010 | 1.24855x | 0.97303x |
| `player/boss/testsubject.rs` | 16,398 | 4,643 | 5,984 | 4,530 | 1.28882x | 0.97566x |

### player/icon_render

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `player/icon_render/sprite_data.rs` | 55,427 | 50,983 | 43,055 | 42,356 | 0.84450x | 0.83079x |
| `player/icon_render/test.rs` | 46,208 | 44,453 | 31,818 | 31,678 | 0.71577x | 0.71262x |

### player/skill

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `player/skill/store.rs` | 28,523 | 7,276 | 7,698 | 7,164 | 1.05800x | 0.98461x |

### player/skill/act

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `player/skill/act/absorb.rs` | 2,724 | 851 | 996 | 824 | 1.17039x | 0.96827x |
| `player/skill/act/accumulate.rs` | 3,950 | 1,121 | 1,347 | 1,062 | 1.20161x | 0.94737x |
| `player/skill/act/assassinate.rs` | 11,075 | 2,888 | 3,535 | 2,819 | 1.22403x | 0.97611x |
| `player/skill/act/berserk.rs` | 5,651 | 1,603 | 1,959 | 1,553 | 1.22208x | 0.96881x |
| `player/skill/act/charge.rs` | 4,128 | 1,168 | 1,430 | 1,136 | 1.22432x | 0.97260x |
| `player/skill/act/charm.rs` | 7,824 | 2,240 | 2,673 | 2,145 | 1.19330x | 0.95759x |
| `player/skill/act/clone.rs` | 11,910 | 3,207 | 3,907 | 3,092 | 1.21827x | 0.96414x |
| `player/skill/act/critical.rs` | 1,715 | 546 | 633 | 525 | 1.15934x | 0.96154x |
| `player/skill/act/curse.rs` | 6,709 | 1,841 | 2,268 | 1,765 | 1.23194x | 0.95872x |
| `player/skill/act/disperse.rs` | 4,158 | 1,167 | 1,411 | 1,124 | 1.20908x | 0.96315x |
| `player/skill/act/exchange.rs` | 5,766 | 1,428 | 1,769 | 1,394 | 1.23880x | 0.97619x |
| `player/skill/act/fire.rs` | 3,162 | 924 | 1,155 | 921 | 1.25000x | 0.99675x |
| `player/skill/act/half.rs` | 4,483 | 1,190 | 1,439 | 1,139 | 1.20924x | 0.95714x |
| `player/skill/act/haste.rs` | 7,194 | 1,890 | 2,331 | 1,834 | 1.23333x | 0.97037x |
| `player/skill/act/heal.rs` | 5,847 | 1,659 | 2,005 | 1,622 | 1.20856x | 0.97770x |
| `player/skill/act/ice.rs` | 6,416 | 1,691 | 2,120 | 1,623 | 1.25370x | 0.95979x |
| `player/skill/act/iron.rs` | 7,272 | 1,906 | 2,350 | 1,852 | 1.23295x | 0.97167x |
| `player/skill/act/minion.rs` | 10,875 | 2,740 | 3,599 | 2,682 | 1.31350x | 0.97883x |
| `player/skill/act/mod.rs` | 1,754 | 787 | 860 | 778 | 1.09276x | 0.98856x |
| `player/skill/act/poison.rs` | 8,159 | 2,245 | 2,759 | 2,162 | 1.22895x | 0.96303x |
| `player/skill/act/possess.rs` | 2,425 | 647 | 813 | 639 | 1.25657x | 0.98764x |
| `player/skill/act/quake.rs` | 3,648 | 993 | 1,218 | 961 | 1.22659x | 0.96777x |
| `player/skill/act/rapid.rs` | 3,168 | 866 | 1,061 | 842 | 1.22517x | 0.97229x |
| `player/skill/act/revive.rs` | 4,062 | 1,168 | 1,410 | 1,139 | 1.20719x | 0.97517x |
| `player/skill/act/shadow.rs` | 4,230 | 1,204 | 1,434 | 1,158 | 1.19103x | 0.96179x |
| `player/skill/act/slow.rs` | 5,549 | 1,496 | 1,844 | 1,451 | 1.23262x | 0.96992x |
| `player/skill/act/summon.rs` | 12,939 | 3,475 | 4,208 | 3,345 | 1.21094x | 0.96259x |
| `player/skill/act/thunder.rs` | 3,610 | 953 | 1,140 | 928 | 1.19622x | 0.97377x |

### player/skill/skl

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `player/skill/skl/corpse.rs` | 664 | 198 | 269 | 196 | 1.35859x | 0.98990x |
| `player/skill/skl/counter.rs` | 3,870 | 1,004 | 1,247 | 980 | 1.24203x | 0.97610x |
| `player/skill/skl/defend.rs` | 2,425 | 640 | 817 | 615 | 1.27656x | 0.96094x |
| `player/skill/skl/hide.rs` | 5,792 | 1,565 | 1,602 | 1,500 | 1.02364x | 0.95847x |
| `player/skill/skl/merge.rs` | 9,384 | 2,038 | 2,622 | 1,999 | 1.28656x | 0.98086x |
| `player/skill/skl/mod.rs` | 957 | 442 | 479 | 438 | 1.08371x | 0.99095x |
| `player/skill/skl/none.rs` | 2,047 | 588 | 765 | 575 | 1.30102x | 0.97789x |
| `player/skill/skl/protect.rs` | 21,963 | 5,194 | 6,402 | 5,063 | 1.23258x | 0.97478x |
| `player/skill/skl/reflect.rs` | 3,855 | 995 | 1,213 | 976 | 1.21910x | 0.98090x |
| `player/skill/skl/reraise.rs` | 1,903 | 666 | 663 | 638 | 0.99550x | 0.95796x |
| `player/skill/skl/shield.rs` | 2,788 | 801 | 985 | 776 | 1.22971x | 0.96879x |
| `player/skill/skl/upgrade.rs` | 4,208 | 1,123 | 1,415 | 1,082 | 1.26002x | 0.96349x |
| `player/skill/skl/zombie.rs` | 4,164 | 1,077 | 1,335 | 1,048 | 1.23955x | 0.97307x |

### player/test

| 文件 | 字符数 | OpenPangu | V4P | GLM 5.2 | V4P / OpenPangu | GLM 5.2 / OpenPangu |
|---|---:|---:|---:|---:|---:|---:|
| `player/test/basic.rs` | 23,588 | 7,176 | 8,393 | 6,803 | 1.16959x | 0.94802x |
| `player/test/boss.rs` | 673 | 205 | 247 | 204 | 1.20488x | 0.99512x |
| `player/test/minions.rs` | 49,100 | 13,482 | 16,425 | 12,715 | 1.21829x | 0.94311x |
| `player/test/shadow_sync.rs` | 5,550 | 1,362 | 1,773 | 1,361 | 1.30176x | 0.99927x |
| `player/test/skill_store.rs` | 8,943 | 2,281 | 2,842 | 2,213 | 1.24594x | 0.97019x |
| `player/test/skills.rs` | 55,152 | 14,408 | 15,372 | 13,856 | 1.06691x | 0.96169x |
| `player/test/states.rs` | 8,694 | 2,324 | 2,928 | 2,256 | 1.25990x | 0.97074x |
| `player/test/weapons.rs` | 1,382 | 463 | 544 | 432 | 1.17495x | 0.93305x |

