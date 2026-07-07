# 题目

CTHR: Constraint-Traceable Hierarchical Rule Compilation for Auditable KG-Driven Optimization

# abstract

在飞行程序设计和建筑设计这类高安全要求工业领域，基于知识的决策支持系统不仅需要返回数值上可行的决策，还需要保证决策与源规则一致，并能够追溯到规则来源。在这些领域中，设计规范容易被组织为知识图谱（KG），但 KG 本身并不会直接给出一个可供优化器使用、且带有来源追溯信息的可行域。针对这一问题，本文提出 Constraint-Traceable Hierarchical Rule Compilation（CTHR），一种保留来源追溯信息的符号建模框架，用于将 KG 中的规则转换为可审计的优化可行域。对于给定的具体设计场景，CTHR 解析规则适用性、依赖、互斥、覆盖、优先级和场景参数传播六类关系，构造对应的合法规则结构，并将每个规则结构编译为精确的可行域，同时标注规则级和文档级来源信息。编译后的可行域随后可以被下游约束优化器使用，并且返回的决策会附带证书，用于溯回KG中的源证据，理论推导表明，CTHR 编译得到的可行域能够保持源规则语义所诱导的约束范围，既不会接纳源规则禁止的决策，也不会排除合法路径的决策。在60个飞行程序设计和建筑设计领域的优化任务上的实验表明，相比平铺规则,CTHR 使落在可行域内的优化决策数量提高了18.4%，并在有效规则恢复、可行域语义一致性、优化结果合法性和证书可追溯性方面保持稳定表现。此外，CTHR还能兼容ASP/clingo、CP-SAT/OR-Tools 和 SCIP这类符号建模和求解方法，生成语义有效、可审计的可行域。




# 1.introduction

第一段：说明知识图谱不能被直接用于设计决策任务

基于知识的决策系统正越来越多地应用于高保障工业流程中，例如飞行程序设计和建筑设计。在这些领域，一个明显趋势是将法规文件、标准和操作手册中的要求组织为知识图谱（KGs）[1, 2]，其中实体、关系、源文档链接和规则标注共同表示领域知识。这类知识图谱有助于规则检索、相关要求关联以及规则来源追溯；然而，它们并无法根据具体的问题从KG检索出的规则中形成决策所需的可行域，并且无法实现决策到源规则的追溯。


第二段：说明知识图谱中的规则并不是平铺的，二是包含多种关系的

在领域知识被编码为知识图谱之后，它仍然需要被转化为可操作的决策机制。在许多实际场景中，这意味着需要将 KG 证据映射为结构化规则库，选择适用于具体场景的规则，编译由此产生的约束，然后求解一个优化问题，这个过程当前主要由具备领域知识和优化算法知识的专家完成。此外，真实的知识图谱不止包含平铺规则。它们往往包含例外规则、跨文档优先级、互斥的替代方案，以及会同时影响多项要求的场景相关条件。将符号知识转换为可行的决策空间，本身就是一个核心的知识工程问题，不是一个简单的预处理步骤。


第三段

大多数约束学习与优化方法假设一种平铺式的规则到约束映射，每条激活规则贡献一个独立不等式，所有这些不等式的合取被视为可行集。然而，面向技术规范的源知识图谱并不是以这种方式运作的。例外规则可能会在特定程序类别下覆盖基础规则；当不同文档发生冲突时，跨文档优先级决定应以哪一来源为准；而替代性模板可能代表不同的合规路径，而不是必须被合并在一起的要求。例如，在建筑规范合规中，建筑物是否配备自动喷水灭火系统，可能会改变最大疏散距离限制、出口宽度阈值以及防火分隔要求。如果建模层忽略这些规则交互，优化器可能会求解一个数值上看似可行、但并不符合源规则语义的区域。同时，这也会丢失解释返回决策为何合规所需的来源追溯信息。

第四段

本文关注源知识图谱与优化器之间的建模层。纯神经约束优化方法 [3, 4] 能够在学习得到或给定的约束下提升目标性能，但它们通常假设可行集本身已经正确，并且不保留规则级来源追溯信息。纯符号推理和基于 SMT 的检查方法 [5] 可以验证合规性，但它们并不能直接提供一个与源规则相链接、且可被下游优化器在不同决策查询中复用的可行域。因此，基于知识的决策系统需要一种能够在保持可行决策范围不变、并保留审计证据的前提下，将知识图谱规则语义依据具体问题转换为优化约束的建模方法。

第五段

为解决上述问题，本文提出 CTHR（Constraint-Traceable Hierarchical Rule Compilation），一种面向 KG 驱动优化、保留来源追溯信息的符号建模框架。在 CTHR 中，源 KG 证据首先被映射为rule library,随后针对每个具体场景解析规则适用性、依赖关系、互斥关系、覆盖关系和优先级关系,剩余的合法规则结构被编译为精确的可行域；并且每个编译后的可行域都保留源文档来源信息。编译得到的可行域随后可以传递给求解器、经典约束优化器或多目标搜索方法。更重要的是，系统返回的不仅是一个数值上可行的解，还包括一份与源规则相链接的审计记录，使领域专家能够将该决策追溯到支撑它的激活规则和源文档。


本文的主要贡献如下：

1.本文提出 Constraint-Traceable Hierarchical Rule Compilation（CTHR），一种用于将 KG 中带来源追溯的规则转换为可审计、可优化可行域的符号建模框架。面向具体设计场景，CTHR 从 grounding 后的候选规则出发，解析规则间的适用、依赖、互斥、覆盖、优先级和参数传播关系，构造合法规则结构，并将其编译为下游求解器可直接使用的 feasible cells。同时，CTHR 在编译过程中保留规则级和文档级 provenance，使优化结果能够追溯至源规则与源文档证据。

2.本文将 KG 规则到优化约束之间的语义缺口形式化为六类核心规则关系：规则适用性、规则依赖、规则互斥、规则覆盖、规则优先级和场景参数传播。基于上述关系，本文设计了 LLM-assisted KG-to-rule-library extraction 流程，从 KG evidence 中自动抽取结构化规则记录，包括规则前提、约束结论、规则关系和来源追溯信息。实验结果表明，不同 LLM 基座均能够支持 KG 到 rule library 的结构化抽取，并在后续 CTHR 建模流程中保持稳定的下游性能。

3.本文构建了一个覆盖飞行程序设计和建筑规范设计两个领域的基准数据集，包含 60 个具体设计优化任务，涵盖常规优化场景和 interaction-rich stress 场景。每个任务均包含 grounded scenario、candidate rules、reference valid rules、valid rule structures、executable reference constraints、optimization objectives 和 source evidence 等信息。数据集经过人工审核，可用于系统评估规则恢复、可行域语义一致性、优化结果合法性和证书可追溯性。

4.实验表明，CTHR 能够从结构化候选规则中恢复正确的合法规则结构，生成与源规则语义一致的可行域，并返回语义合法的优化结果。CTHR 使落在可行域内的优化决策数量提高了18.4%，并在有效规则恢复、可行域语义一致性、优化结果合法性和证书可追溯性方面保持稳定表现。此外，CTHR还能兼容ASP/clingo、CP-SAT/OR-Tools 和 SCIP这类符号建模和求解方法，生成语义有效、可审计的可行域。






# 2.Related work


# 3. Motivation
在很多高安全要求的工业场景中，工程师真正面对的并不是“有没有知识”的问题，而是“如何把知识用起来”的问题。以飞行程序设计和建筑规范设计为例，领域规范可以被整理成知识图谱，规则、条文、条件和来源都可以被很好地存储和检索。但当工程师要解决一个具体设计问题时，KG 通常只能告诉他相关规则有哪些，却不能直接告诉优化器：哪些变量可以取什么值，哪些规则在当前场景下生效，哪些规则因为例外、互斥或优先级应该被排除，以及最终的可行域到底是什么。现实中，这一步往往依赖少数既懂业务规范、又懂优化建模的专家手工完成。本文的出发点就是尝试缩小这道鸿沟：让 KG 中的规则不再只停留在“可查询的知识”，而是能够在具体设计场景下被转换为可执行、可审计、可优化的决策可行域。

另一方面，LLM令人震惊的发展速度为工业知识建模提供了新的可能。它们在阅读规范文本、理解业务术语、归纳规则结构以及整合多来源知识方面表现出较强能力，使得从 KG evidence 中抽取结构化规则库变得更加现实。然而，在航空和建筑这类高安全要求领域，我们认为最终决策不宜直接依赖概率生成模型给出，而应由确定性、可验证、可审计的机制产生。更合理的方式是让 LLM 承担其更擅长的角色：辅助理解文本、组织知识并生成候选规则；而将最终的规则解析、约束编译、可行性验证和证书生成交由形式化符号框架完成。这一思路也与 LLM-Modulo planning 相关工作的观点一致：LLM 可以帮助生成候选方案和组织知识，但规划、验证与执行仍应由外部形式化模块承担。CTHR 正是基于这一思想设计的：利用 LLM 和 KG 辅助规则库构建，同时通过可追溯的符号规则编译框架保证最终决策的可验证性和可审计性。


# 4. Problem Formulation



# 5. Method

\CTHR{} 框架建立了一个自动化的端到端知识工程流水线，能够从原始的技术规范文件提取出针对具体问题的、面向优化的可行域空间。如图1所示，CTHR方法包含知识Knowledge Acquisition、CTHR Modeling Layer、Solver backends、Optimized Decision& Audit Certificate四部分。Knowledge Acquisition部分主要完成知识图谱和rule librarya的构建，CTHR Modeling Layer根据具体问题从rule library中提取和编译出可行域，Solver backends依据可行域具体问题的目标和偏好进行求解，Optimized Decision& Audit Certificate得到的求解的结果以及对应的源文件

## 5.1 Knowledge Acquisition


Knowledge Acquisition是将技术规范文档转换为可计算、可追溯的规则知识表示。CTHR框架使用 Cognee 构建源链接知识图谱。该过程将原始 PDF 解析为文档片段，并为每个片段保留文档名称、页码等来源信息。随后，Cognee 对文本片段中的规范对象、适用条件、数值阈值、单位、设计对象、约束关系和跨条款引用进行结构化抽取，并将其组织为知识图谱中的节点和边。由此得到的知识图谱不仅保存规范文本内容，还保存规范证据与原始文档之间的 provenance links，使后续生成的规则能够追溯到具体 source document、clause 或 evidence。

在构建完成 KG 之后，CTHR 使用 LLM 将 KG转换为结构化 rule library。LLM 作为 KG-to-rule extractor，将KG 证据转换为统一格式的 rule records，每个 rule records 中包含源文本片段、文档标识、章节或页码提示、相关实体节点、数值和单位信息，以及与当前片段相邻的 KG 关系。

提取LLM 提示词由四类内容组成：第一，任务说明，要求模型仅基于给定 KG 证据抽取技术规范规则；第二，输入证据，包括 source chunk、相关实体、数值、单位和 KG 邻接信息；第三，目标输出 schema，规定每条规则必须包含 rule identifier、rule name、domain、rule type、applicability guard、consequent constraints、parameters、inter-rule relations 和 provenance fields；第四，格式约束，要求输出为可解析 JSON，并禁止生成没有来源证据支持的阈值、单位或规则关系。

生成后的 rule records 会经过自动校验和规范化处理，以确保其 JSON 格式、规则标识、约束表达式、单位、数值来源和 provenance 信息均满足后续符号建模要求。校验后的规则被汇总为 rule library
\[
\mathcal{R}=\{r_i\}_{i=1}^{N}.
\]每条规则 \(r_i\) 

包含两类信息：一类是可执行语义信息，例如适用条件、参数、约束表达式和规则类型；另一类是结构和审计信息，例如 dependency、exclusion、override、precedence、parameter propagation 关系，以及文档级和规则级 provenance。


## 5.2 CTHR Modeling Layer



### 5.2.1 candidate rule grounding

第一步是依据具体任务题目从rule library中生成出candidate rule  这里需要补充筛选细节


Candidate rule grounding 的目标，是针对一个具体优化任务 \(q\)，从 rule library \(\mathcal{R}\) 中生成一个高召回的候选规则集合 \(C(q)\)。具体而言，CTHR 对 rule library 中的每条规则 \(r_i\) 计算一个 task-grounding score。该分数综合考虑以下匹配项：任务文本与规则名称、规则 ID、规则类型和 provenance section 的词项重叠；规则适用条件中的字段和值是否能够与当前场景事实匹配；规则约束中的变量是否能够映射到任务中的决策变量；规则使用的单位是否与任务变量单位一致；以及规则来源文档或来源领域是否与当前任务领域一致。对于带有明确适用条件的规则，如果该适用条件在当前场景下被满足，则获得额外权重。该过程可以概括为：
\[
s(r_i,q)=
s_{\mathrm{text}}+
s_{\mathrm{guard}}+
s_{\mathrm{var}}+
s_{\mathrm{unit}}+
s_{\mathrm{prov}}.
\]

其中，\(s_{\mathrm{text}}\) 由词项重叠计算得到：系统将任务可见字段和规则元数据归一化为 token set，并计算二者交集大小。归一化过程包括小写化、按下划线、连字符和标点切分、去除通用停用词，并对少量领域缩写进行同义扩展。其余分数项则分别由 guard 满足性、变量绑定、单位一致性和来源文档匹配给出。因此，该 grounding score 是确定性的符号匹配分数，而不是基于 embedding 的向量相似度。

随后，系统按照 grounding score 对规则排序，并保留得分较高的规则作为初始候选。candidate rule grounding 的输出不是最终有效规则，而是一个 intentionally broad candidate set：
\[
C(q)=\{r_i\in\mathcal{R}\mid s(r_i,q)\ge \tau \ \text{or}\ r_i\ \text{passes recall guard}\}.
\]
这个候选集通常明显大于最终 valid rule set。

### 5.2.2 Rule-Interaction Resolution

Candidate rule grounding 以高召回为目标，因此得到的候选规则集合 \(C(q)\) 通常明显大于当前任务真正生效的规则集合。候选规则中既包含当前场景下应当生效的规则，也包含仅因任务词项、变量名称或来源文档相近而被召回的无关规则；同时，一些规则本身并不直接形成优化约束，但会通过依赖、例外覆盖、互斥、优先级或参数传播关系影响最终有效规则结构。因此，CTHR 在 candidate grounding 之后执行 Rule-Interaction Resolution，将 broad candidate set 解析为当前任务的 valid rule set：
\[
V(q)\subseteq C(q).
\]该过程如 Algorithm~\ref{alg:rule_interaction_resolution} 所示。其核心思想是先通过任务 profile 对候选规则进行约束化收缩，再使用可选的 LLM-assisted filtering 去除明显语义不一致的候选规则，最后通过符号规则关系解析得到当前任务下真正生效的 valid rules。
latex



\begin{algorithm}[t]
\caption{Rule-Interaction Resolution}
\label{alg:rule_interaction_resolution}
\KwIn{Task $q$; broad candidate set $C(q)$; rule relation graph $\mathcal{E}_{\mathrm{rel}}$; optional LLM filter $\mathcal{M}$}
\KwOut{Valid rule set $V(q)$}

Construct task profile
$P(q)=(G_q^{\mathrm{req}},G_q^{\mathrm{allow}},G_q^{\mathrm{block}},\sigma_q)$\;

$C_{\mathrm{seed}}(q)\leftarrow \emptyset$\;

\ForEach{$r\in C(q)$}{
    Map rule $r$ to rule groups $G(r)$\;

    Compute profile indicators
    $I_{\mathrm{req}}, I_{\mathrm{allow}}, I_{\mathrm{block}}, I_{\mathrm{unmatch}}$\;

    Compute visible binding score $v(r,q)$ from guard fields,
    decision variables, units, and normalized tokens\;

    Compute profile score:
    \[
    s_{\mathrm{prof}}(r,q)=
    w_{\mathrm{req}}I_{\mathrm{req}}
    +w_{\mathrm{allow}}I_{\mathrm{allow}}
    -w_{\mathrm{block}}I_{\mathrm{block}}
    -w_{\mathrm{unmatch}}I_{\mathrm{unmatch}}
    +v(r,q)
    \]\;

    \If{$s_{\mathrm{prof}}(r,q)\ge \tau_{\mathrm{seed}}$}{
        $C_{\mathrm{seed}}(q)\leftarrow C_{\mathrm{seed}}(q)\cup\{r\}$\;
    }
}

Remove blocked or unmatched rules from $C_{\mathrm{seed}}(q)$\;
Re-inject required rules that are not blocked\;
Obtain $C_{\mathrm{prof}}(q)$\;

\If{LLM-assisted filtering is enabled}{
    $C_{\mathrm{prof}}(q)\leftarrow \mathcal{M}(q,C_{\mathrm{prof}}(q))$\;
}

$V(q)\leftarrow C_{\mathrm{prof}}(q)$\;

Remove rules whose applicability guards are false under $q$\;
Close $V(q)$ under dependency and parameter/formula propagation\;
Resolve exclusion, override, and precedence conflicts in $V(q)$\;

\Return{$V(q)$}\;
\end{algorithm}

在 Algorithm~\ref{alg:rule_interaction_resolution} 的前半部分，CTHR 首先构造任务 profile。对于给定任务 \(q\)，任务 profile 表示为：
\[
P(q)=\left(G_q^{\mathrm{req}},G_q^{\mathrm{allow}},G_q^{\mathrm{block}},\sigma_q\right),
\]其中 \(G_q^{\mathrm{req}}\)、\(G_q^{\mathrm{allow}}\) 和 \(G_q^{\mathrm{block}}\) 分别表示当前任务显式需要、允许保留和应当排除的规则组标签，\(\sigma_q\) 表示是否采用严格过滤模式。每条候选规则 \(r\) 也会根据规则标识、规则名称、规则类型、来源领域、适用条件、约束变量和 provenance 信息映射到一个或多个规则组 \(G(r)\)。
为了控制候选规则收缩过程，CTHR 将每条候选规则相对于当前任务划分为 required、allowed、blocked 和 unmatched 四类。Required rules 表示规则组被当前任务类型、场景事实或决策变量显式触发；allowed rules 表示规则虽非核心触发规则，但与当前任务领域、设计对象或结构关系一致；blocked rules 表示规则组与当前任务的场景条件、设计对象或来源领域冲突；unmatched rules 表示规则既未命中 required 或 allowed profile，也缺少明确结构关系支持。这四类状态由如下指示变量给出：
\[
I_{\mathrm{req}}(r,q)=\mathbf{1}\left[G(r)\cap G_q^{\mathrm{req}}\neq\emptyset\right],
\]\[
I_{\mathrm{allow}}(r,q)=\mathbf{1}\left[G(r)\cap G_q^{\mathrm{allow}}\neq\emptyset\right],
\]\[
I_{\mathrm{block}}(r,q)=\mathbf{1}\left[(G(r)\cap G_q^{\mathrm{block}})\setminus G_q^{\mathrm{req}}\neq\emptyset\right],
\]\[
I_{\mathrm{unmatch}}(r,q)=
\sigma_q\cdot
\mathbf{1}\left[
I_{\mathrm{req}}(r,q)=0
\land
I_{\mathrm{allow}}(r,q)=0
\right].
\]这些状态进入 profile score。实现中，required 和 allowed 分别给予较强和较弱的正权重，blocked 和 unmatched 分别给予较强和较弱的负权重。例如在 strict-profile 模式下，required 规则获得 \(+7\) 的 profile bonus，allowed 规则获得 \(+3\)，blocked 规则受到 \(-9\) 的惩罚，unmatched 规则受到 \(-5\) 的惩罚。可见字段一致性项 \(v(r,q)\) 进一步考虑适用条件字段匹配、变量绑定、单位一致性和词项重叠：
\[
v(r,q)=
2.5|F(r)\cap F(q)|
+(3+m_X(r,q))\mathbf{1}[m_X(r,q)>0]
+0.8\mathbf{1}[U(r)\cap U(q)\neq\emptyset]
+\min(3,\lambda_t |T(r)\cap T(q)|).
\]其中 \(F(\cdot)\) 表示场景事实或规则适用条件字段集合，\(m_X(r,q)\) 表示规则变量与任务决策变量之间的可绑定数量，\(U(\cdot)\) 表示单位集合，\(T(\cdot)\) 表示归一化词项集合。
分数筛选之后，CTHR 进一步执行 hard profile filtering：blocked 和 unmatched 规则会从 seed 集合中移除，而命中 required profile 且未被 blocked 的规则会被重新注入。随后，可选的 LLM-assisted filtering 只根据公开任务描述、场景事实、决策变量、候选规则文本、适用条件和 provenance 信息进行辅助过滤或重排序，不使用 reference valid rules、求解结果或语义评价标签。
最后，CTHR 对 \(C_{\mathrm{prof}}(q)\) 执行符号规则关系解析。系统首先移除在当前场景下适用条件明确为假的规则；然后沿 dependency 和 parameter/formula propagation 关系补齐必要的前置规则、公式规则和参数定义规则；最后根据 exclusion、override 和 precedence 关系处理互斥分支、例外覆盖和优先级冲突。由此得到当前任务的 valid rule set：
\[
V(q)=
\mathrm{Resolve}
\left(
C_{\mathrm{prof}}(q),
\mathcal{E}_{\mathrm{rel}},
\phi_q
\right),
\]其中 \(\mathcal{E}_{\mathrm{rel}}\) 表示 rule library 中的规则关系边，\(\phi_q\) 表示当前任务的场景事实、变量绑定和单位信息。Rule-Interaction Resolution 因此将高召回但含噪声的候选规则集合转化为结构化、可追溯、且与当前任务场景一致的 valid rule set，为后续 valid rule structure 构造和可行域编译提供输入。


### 5.2.3 valid rule structure construction

Rule-Interaction Resolution 得到的是当前任务下生效的 valid rule set \(V(q)\)。然而，\(V(q)\) 仍然只是一个规则集合，而不是可以直接平铺编译的约束模型。在技术规范中，一组 valid rules 可能同时包含通用规则、例外规则、参数定义规则、公式规则，以及多个替代合规路径。例如，两个替代模板可以同时被识别为当前任务相关，但它们并不应被合并为同一个约束集合；又如，一条具体约束规则可能依赖若干前置定义或场景参数，这些支撑规则必须与该约束规则保持在同一结构中。因此，CTHR 在 valid rule recovery 之后进一步构造 valid rule structures，用于表示当前任务下一个或多个自洽的合规路径：
\[
\mathcal{S}(q)=\{S_1,S_2,\ldots,S_{K(q)}\}.
\]每个 \(S_k\) 对应一个可独立编译的规则结构，并将在下一步被转换为一个 feasible cell。
具体而言，CTHR 首先在 \(V(q)\) 上构造诱导规则关系图：
\[
H_q=(V(q),\mathcal{E}_{\mathrm{rel}}|_{V(q)}),
\]其中 \(\mathcal{E}_{\mathrm{rel}}|_{V(q)}\) 表示 rule library 中连接 valid rules 的关系边。该图保留了规则之间的依赖、互斥、例外覆盖、优先级和参数传播关系。基于该关系图，CTHR 将 valid rules 分为两类：一类是 core rules，即在当前场景下必须保留、且不属于互斥或替代分支的规则；另一类是 branch rules，即由 alternative、exclusion、override 或 precedence 关系形成的候选分支规则。
随后，CTHR 根据互斥、替代、覆盖和优先级关系识别 conflict classes：
\[
\mathcal{B}(q)=\{B_1,B_2,\ldots,B_m\}.
\]每个 \(B_j\) 表示一组不能同时出现在同一 valid rule structure 中的规则或规则组合。对于每个 conflict class，CTHR 根据当前任务场景、规则适用条件、override 关系和 precedence order 选择可保留的合法分支。如果某个冲突已经由 override 或 precedence 决定，则被覆盖或低优先级规则不会进入候选结构；如果多个替代分支在当前任务下均合法，则它们被保留为不同的 valid rule structures，而不是被合并。
在确定分支之后，CTHR 对每个候选结构执行依赖闭包和参数传播。对于任一候选结构 \(S_k\)，如果其中某条规则 \(r\) 依赖于前置规则、定义规则、参数规则或公式规则，则这些支撑规则必须被纳入同一结构：
\[
\forall r\in S_k,\quad
\mathrm{Dep}(r)\cap V(q)\subseteq S_k.
\]同时，场景相关参数会根据当前任务事实被实例化，并传播到依赖这些参数的规则约束中。若某个候选结构存在无法满足的适用条件、缺失依赖，或包含互斥规则，则该结构被丢弃。
由此得到的每个 valid rule structure 可表示为：
\[
S_k=(R_k,E_k,\gamma_k,\Pi_k),
\]其中 \(R_k\subseteq V(q)\) 是结构中的规则集合，\(E_k\) 是规则之间的关系子图，\(\gamma_k\) 是该结构的场景激活条件，\(\Pi_k\) 是规则级和文档级 provenance 映射。结构 \(S_k\) 必须满足两个基本条件：第一，它是 dependency-closed，即规则所需的依赖、公式和参数定义均被包含；第二，它是 exclusion-consistent，即同一结构中不存在互斥规则或已被覆盖的规则。
因此，Valid Rule Structure Construction 的作用，是将一个有效但仍然平铺的规则集合 \(V(q)\) 转换为一个结构化的合规路径集合 \(\mathcal{S}(q)\)。后续 feasible-region compilation 不直接编译整个 \(V(q)\)，而是分别编译每个 \(S_k\)。这种设计保留了源规则中的析取语义：一个优化决策只需要满足某一个完整的 valid rule structure，即可被视为语义合法；而不需要同时满足所有互斥或替代规则路径。

### 5.2.4 feasible-region compilation

Valid rule structure construction 得到的是一组结构化合规路径
\[
\mathcal{S}(q)=\{S_1,S_2,\ldots,S_{K(q)}\}.
\]每个 \(S_k\) 已经保证在规则层面是依赖闭合且互斥一致的，但它仍然不是优化器可以直接使用的可行域。因此，CTHR 的第四步是将每个 valid rule structure 编译为一个可执行的 feasible cell，并将所有 feasible cells 组合为当前任务的可行域。
具体而言，对于每条规则 \(r\)，CTHR 在 rule library 中维护其对应的 compiled rule-to-constraint templates：
\[
\mathcal{T}(r)=\{t_{r,1},t_{r,2},\ldots\}.
\]每个 template \(t_{r,j}\) 包含约束表达式、所需变量符号、参数符号、单位信息、约束角色以及 provenance 映射。该 template 并不是针对某个测试题临时生成的自然语言解释，而是从 source-rule executable semantics 中抽取得到的结构化约束模板。编译时，CTHR 根据当前任务的决策变量、变量边界、单位系统和场景参数，对 template 中的符号进行绑定和实例化。
对于一个 valid rule structure \(S_k=(R_k,E_k,\gamma_k,\Pi_k)\)，CTHR 收集其中所有规则的约束模板，并执行三类操作。第一，系统检查每个 template 的 required symbols 是否能够由当前任务的决策变量、场景事实或参数传播结果提供；若必要符号无法解析，则该 template 不会被发射为可执行约束。第二，系统对变量名称、单位和参数进行规范化绑定，将规则中的符号边界转换为当前任务变量空间中的表达式。第三，系统将已经实例化的规则约束与任务自身的变量边界合并，形成该结构对应的 feasible cell：
\[
P_k(q)=
\left\{
x\in \mathcal{X}_q
\mid
\bigwedge_{r\in R_k}
\bigwedge_{c\in \Gamma(r,q)}
c(x;\eta_q)\ \text{holds}
\right\},
\]其中 \(\mathcal{X}_q\) 表示任务给定的基础变量域，\(\Gamma(r,q)\) 表示规则 \(r\) 在任务 \(q\) 下实例化后生成的可执行约束集合，\(\eta_q\) 表示由场景事实传播得到的参数取值。
由于不同 valid rule structures 可能对应不同的合规路径，CTHR 不会将所有 \(P_k(q)\) 强行合并为一个平铺合取约束。相反，当前任务的整体可行域被表示为 feasible cells 的析取并集：
\[
F_{\mathrm{CTHR}}(q)=
\bigcup_{k=1}^{K(q)} P_k(q).
\]因此，一个优化决策 \(x\) 只要满足某一个完整 cell \(P_k(q)\)，就被视为满足源规则诱导的可行域语义。这一点对于包含替代模板、互斥规则或例外路径的技术规范尤其重要，因为 flat compilation 会错误地要求一个解同时满足多个本应分离的合规路径。
每个 compiled cell 同时保留可审计元数据：
\[
\mathrm{cell}_k=
(\mathrm{cell\_id}, R_k, \Gamma_k, \gamma_k, \Pi_k, \mathcal{B}_k),
\]其中 \(\mathrm{cell\_id}\) 是可行域单元标识，\(R_k\) 是该 cell 使用的规则集合，\(\Gamma_k\) 是实例化后的可执行约束集合，\(\gamma_k\) 是结构激活条件，\(\Pi_k\) 是规则级和文档级 provenance，\(\mathcal{B}_k\) 是变量绑定和参数实例化记录。后续求解器返回某个解时，CTHR 可以根据其满足的 active cell 生成 certificate，将优化结果追溯到对应的 valid rule structure、规则模板和源文档证据。
编译后的 feasible cells 与具体求解器解耦。CTHR 输出的是统一的约束表示，随后可以由默认优化器、ASP/clingo、CP-SAT/OR-Tools 或 SCIP 等后端消费。不同求解器只负责在同一组 compiled cells 上搜索优化解，而不重新决定哪些规则有效，也不重新解释规则来源。这样，CTHR 将符号规则解析与数值优化执行分离：前者保证可行域语义和来源追溯，后者负责在该可行域内完成目标优化。
因此，Feasible-Region Compilation 的作用，是把来源可追溯的 valid rule structures 转换为求解器可执行的几何可行域：
\[
\mathcal{S}(q)
\longrightarrow
\{P_1(q),P_2(q),\ldots,P_{K(q)}(q)\}
\longrightarrow
F_{\mathrm{CTHR}}(q).
\]这一过程确保 CTHR 输出的不是一组松散规则，而是可直接用于优化、同时保留规则链和文档证据的 auditable feasible region。


## 5.3 Solver Backends
经过 CTHR Modeling Layer 之后，系统输出的不是绑定到某一个特定求解器的模型，而是一组已经编译好的 feasible cells。也就是说，CTHR 负责完成规则筛选、规则关系解析、valid rule structure 构造和可行域编译；求解器只负责在该可行域上执行优化。对于任务 \(q\)，CTHR 生成的可行域可以表示为：
\[
F_{\mathrm{CTHR}}(q)=\bigcup_{k=1}^{K(q)}P_k(q),
\]其中每个 \(P_k(q)\) 对应一个 valid rule structure 编译得到的 feasible cell。下游求解器的目标是在该可行域内求解：
\[
\hat{x}_q \in \arg\min_{x\in F_{\mathrm{CTHR}}(q)} g_q(x),
\]其中 \(g_q(x)\) 表示当前任务的目标函数或标量化后的多目标函数。
需要强调的是，求解器后端不重新决定哪些规则有效，也不重新解析规则之间的依赖、互斥、覆盖或优先级关系。这些语义操作已经在 CTHR Modeling Layer 中完成。求解器接收到的是已经确定的 feasible cells、变量边界、约束表达式和目标函数。因此，Solver Backends 在整个框架中的角色是执行层，而不是规则解释层。
不过，不同求解器对同一组 feasible cells 的符号表示方式不同。对于 ASP/clingo 后端，CTHR 需要将规则结构和 cell membership 编码为逻辑事实、选择规则和完整性约束；对于 CP-SAT/OR-Tools 后端，约束通常需要被转换为整数缩放后的线性约束、布尔变量和指示变量；对于 SCIP 后端，约束可以根据表达式类型被表示为连续、整数、线性、非线性或混合整数约束。因此，CTHR 输出的是统一的可行域语义，而后端适配器负责将同一组 compiled cells 转换为对应求解器可接受的建模语言。
这种设计使 CTHR 具有 solver-agnostic 的特性。默认优化器、ASP/clingo、CP-SAT/OR-Tools 和 SCIP 都可以在同一组 CTHR compiled cells 上求解，只是它们使用不同的符号编码和搜索机制。求解器的输出包括优化解、目标值以及被满足或被选中的 active cell。随后，CTHR 根据 active cell 将求解结果映射回对应的 valid rule structure、规则集合、约束模板和 source provenance。
因此，Solver Backends 的作用是将 CTHR 已经编译好的可审计可行域交给不同优化或符号求解系统执行。CTHR 保证可行域语义、规则结构和来源追溯的一致性；求解器负责在该可行域中寻找满足目标函数的最优或近似最优解。这样，规则语义编译与数值/符号优化执行被明确解耦，既保证了方法的可审计性，也增强了框架对不同求解器的兼容性。

## 5.4 Optimizaed Decision& Audit Certificate
在 Solver Backends 得到优化结果之后，CTHR 不仅返回数值解，还会生成与该解对应的 audit certificate。该 certificate 的目的，是说明优化解为什么被认为是合法的、它满足了哪一个 feasible cell、该 cell 来自哪一个 valid rule structure，以及这些规则最终能够追溯到哪些源文档证据。因此，CTHR 的最终输出可以表示为：
\[
o_q=(\hat{x}_q, g_q(\hat{x}_q), \mathrm{cell}_k, \mathrm{cert}_q),
\]其中 \(\hat{x}_q\) 是求解器返回的优化决策，\(g_q(\hat{x}_q)\) 是目标函数值，\(\mathrm{cell}_k\) 是该解所属的 active cell，\(\mathrm{cert}_q\) 是对应的审计证书。
具体而言，CTHR 首先检查求解器返回的 \(\hat{x}_q\) 属于哪一个 compiled feasible cell：
\[
\hat{x}_q \in P_k(q).
\]一旦确定 active cell，系统即可恢复该 cell 对应的 valid rule structure \(S_k\)。由于每个 compiled cell 在构造时都保存了 cell identifier、规则集合 \(R_k\)、实例化约束 \(\Gamma_k\)、变量绑定记录 \(\mathcal{B}_k\)、结构激活条件 \(\gamma_k\) 和 provenance 映射 \(\Pi_k\)，因此 CTHR 可以从优化解反向追踪到完整的规则链：
\[
\hat{x}_q
\rightarrow
P_k(q)
\rightarrow
S_k
\rightarrow
R_k
\rightarrow
\mathrm{Provenance}(R_k).
\]Audit certificate 主要包含四类信息。第一，decision summary，包括优化变量取值、目标函数值、求解器后端和 active cell identifier。第二，constraint satisfaction record，包括该解满足的 instantiated constraints，以及这些约束来自哪些 compiled rule-to-constraint templates。第三，valid rule chain，包括 active cell 对应的 valid rule structure 中的规则 ID、规则名称、规则类型、依赖关系、覆盖关系和优先级关系。第四，source provenance，包括每条规则在 rule library 中记录的 source document、page number、section 或 clause identifier、evidence chunk，以及 KG 中对应的实体或关系节点。
追溯过程依赖于 CTHR 在前面各阶段保留的 provenance 映射。Knowledge Acquisition 阶段中，Cognee 为每个文档片段保留 source document、page、section 和 evidence chunk 等元数据；KG-to-rule extraction 阶段中，LLM 生成的 rule records 必须包含 provenance fields；feasible-region compilation 阶段中，每个约束模板都会保存其来源规则和原始证据链接。因此，当求解器返回一个解时，CTHR 不需要重新解释文本，也不需要向 LLM 询问解释，而是沿着已经保存的映射关系进行确定性回溯。
形式上，若某个 compiled constraint \(c\in\Gamma_k\) 来自规则 \(r\)，而规则 \(r\) 的 provenance 为：
\[
\Pi(r)=
(\mathrm{doc},\mathrm{page},\mathrm{section},\mathrm{chunk},\mathrm{kg\_nodes}),
\]则 certificate 中会记录：
\[
c \rightarrow r \rightarrow \Pi(r).
\]对于一个完整的 active cell，certificate 会列出该 cell 中所有规则的 provenance：
\[
\mathrm{cert}_q =
\left\{
(\,c,r,\Pi(r)\,)
\mid
c\in\Gamma_k,\ r\in R_k
\right\}.
\]这样，领域专家可以从最终优化解逐层追踪到可执行约束、valid rule structure、规则 ID、规则文本片段以及原始规范文档位置。
因此，Optimized Decision and Audit Certificate 的作用，是把求解器返回的数值结果重新连接回源规则证据。求解器负责找到一个满足 compiled feasible region 的优化解，而 CTHR certificate 负责回答“该解为什么合规”“它满足的是哪条规则路径”“每条规则来自哪个源文档证据”。这使得 CTHR 不仅生成可优化的可行域，也生成可审计、可解释、可追溯的决策结果。



# 6. Theoretical Analysis

# 7. Experiment


## 7.1 Experimental Setup

1.说明航空KG和建筑KG的原始数据是啥

2.说明航空测试集和建筑测试集
总体：数量，rule library的数量，
每个题：题型包含的参数和含义，哪些是输入，哪些是参考答案；每道题六种关系的覆盖关系(可放附录)

其中航空领域的测试题经过行业专家的审核，建筑领域测试题的审核通过GPT Claude Gemini对话10轮和人工审核得到

3.KG转到rule library调用的模型





4.评价指标的计算方式


每个题grounding筛选之后的rule和valid rule的比例放在专门证明6种关系检索的实验章节

## 7.2 总体性能评价

flat方法
CTHR默认建模和求解方法
ASP建模和clingo求解方法
CP-SAT符号建模和OR-Tools求解方法
SCIP求解方法

包含两个实验结果表格：
1.显示是完整KG-to-constraint modeling pipeline baseline：从 candidate rules 出发，自己生成可行域并求解。
2.使用 CTHR 已经生成的 compiled cells，再换不同求解器。

评价指标：
针对数据集整体的指标

第一个表格：
Rule Precision，平均规则精确率，每个题方法筛选出的 valid rules 里，有多少是真的 reference valid rules，然后求平均；
Rule Recall，平均规则召回率，reference valid rules 里，有多少被方法找到了，也是先算每个题，然后求平均；
Formal Constraint Satisfaction Rate，形式约束满足率，每个方法返回解是不是在自己的可行域内，在可行域内的题目数量除以总的题目数量；
Semantic Constraint Satisfaction Rate，语义约束满足率，每个题方法返回的优化解，是否满足 source-rule reference semantics， 满足的题目数量除以总的题目数量；
False accept， 方法认为自己的解可行，但 source-rule reference 判断它非法，不非法的题目总数/总的题目数；
Invalid cases， 非法优化案例数， 在全部任务中，最终优化解不满足 source-rule reference semantics 的任务数/总的任务数




第二个表格：
Solve 求解成功率，每个方法返回优化值的题目数/总的题目数；
Cell CSR, 求解器返回的解是否满足 CTHR compiled cells, 满足的题数/总题目数；
Objective gap，相对最优目标差距，每个方法每个题相比最佳优化目标的差值的百分比，然后评价



## 7.3 candidate到valid rule实验

证明CTHR方法能够从cadidate中依据六个规则的关系筛选出合适的 valid rule

以CTHR为默认方法

包含一个表格，除了表头每行是一个题，可能放在正文选10个题出来，完整的题目放在附录

评价指标：
针对每个题

Candidate / Reference Ratio	候选规则数 / 参考 valid rule 数	衡量输入候选集比最终 valid rules 宽多少
Predicted / Reference Ratio	方法筛出的规则数 / 参考 valid rule 数	看方法最终筛出的规则数量是否接近 reference
Rule-ID Precision	规则精确率	方法筛出的规则里，有多少是真的 valid rules



## 7.4 KG-TO-RULE LIBRARY实验

比较qwen,deepseek pro生成的rule library格式的正确性和效果

包含两个实验结果表格：
1.三个api形成rule library建模语义的正确性，rule的数量
2.三种rule library在CTHR默认建模和求解方法下的正确性

评价指标：
评价rule library
1.
Rules 规则数量  该 API 生成的 rule record 总数
rovenance valid 来源追溯正确率 每条 rule 是否能追溯到正确 source document / clause / evidence chunk，能成功追溯占总rule数的比例


2.
评价的也是整个数据集，默认的library结果来自于6.2的表1

Rule Precision，平均规则精确率，每个题方法筛选出的 valid rules 里，有多少是真的 reference valid rules，然后求平均；
Rule Recall，平均规则召回率，reference valid rules 里，有多少被方法找到了，也是先算每个题，然后求平均；
Formal Constraint Satisfaction Rate，形式约束满足率，每个方法返回解是不是在自己的可行域内，在可行域内的题目数量除以总的题目数量；
Semantic Constraint Satisfaction Rate，语义约束满足率，每个题方法返回的优化解，是否满足 source-rule reference semantics， 满足的题目数量除以总的题目数量；
False accept， 方法认为自己的解可行，但 source-rule reference 判断它非法，不非法的题目总数/总的题目数；
Invalid cases， 非法优化案例数， 在全部任务中，最终优化解不满足 source-rule reference semantics 的任务数/总的任务数



## 7.5 审计可追溯性实验

比较CTHR默认建模和求解方法，ASP建模和clingo求解方法，SMT符号建模和Z3求解方法，MILP符号建模和HiGHS求解方法可审计性实验的实验结果，主要是说明这些本身不支持可追溯性的方法通过CTHR框架后可以达到审计可追溯的目的



包含1个表格

评价指标
针对的是整个数据集统计


Certificate coverage	证书覆盖率	有多少返回的优化解附带 certificate；
Rule provenance valid	规则来源有效率	certificate 中每条 rule 是否能在 KG / rule library 中找到对应 provenance，cetificate中的rule再在rule library中找到记为1，不能找到记为0，找到的/certificate中总的rule数；
Valid-chain trace complete，合法规则链追溯完整率，certificate 是否列出当前解所属的完整 valid rule structure，完整的题目数/总的题目数





## 7.6 消融实验

比较默认CTHR方法，如果去掉了6类关系中的某一种，两个数据集的求解情况

一个表格

评价指标：
针对整个数据集


成功求解数和总问题数的比例；
Formal CSR	形式约束满足率	返回点是否满足该方法自己构造的约束，满足的题目数除以总的题目数；
Semantic CSR / Sem-CSR	语义约束满足率	返回点是否满足 source-rule reference semantics，满足的题目数除以总的题目数；
Rule-ID Precision	规则精确率	ablation variant 筛出的 valid rules 中，有多少是真正 reference valid rules，真正符合的题目数/总题目数；
Rule-ID Recall	规则召回率	reference valid rules 中，有多少被 ablation variant 找到了，真正找到的题目数/总的题目数。


# 8. Conclusion
