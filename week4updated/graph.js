// graph.js
// Generic force-directed graph renderer using D3.js.
//
// Usage:
//   1. Ensure D3 is loaded in index.html, e.g.:
//        <script src="https://d3js.org/d3.v7.min.js"></script>
//   2. Add a container element:
//        <div id="graph" style="width: 800px; height: 600px;"></div>
//   3. Include this script after D3.
//   4. Create and render a graph:
//        const graphRenderer = new GraphRenderer('graph');
//        const graph = {
//          nodes: [{ id: 'u_1' }, { id: 'i_42' }, ...],
//          edges: [{ source: 'u_1', target: 'i_42' }, ...]
//        };
//        const scores = { 'u_1': 0.2, 'i_42': 0.9 }; // optional
//        graphRenderer.renderGraph(graph, scores);
//
// Optional interaction with app.js:
//   - On node click, this class will call window.app.selectNode(nodeId)
//     if such a function exists. You can hook this to drive RAG / RecSys
//     introspection from the graph UI.

class GraphRenderer {
    /**
     * @param {string} containerId  DOM id of a block-level element where
     *                              the SVG should be created.
     * @param {Object} options      Extra settings (optional).
     *        { minRadius, maxRadius, defaultRadius }
     */
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.svg = null;
        this.zoomGroup = null;
        this.simulation = null;

        this.nodes = [];
        this.links = [];
        this.width = 0;
        this.height = 0;
        this.selectedNodeId = null;

        this.options = {
            minRadius: options.minRadius || 6,
            maxRadius: options.maxRadius || 22,
            defaultRadius: options.defaultRadius || 10
        };

        this._initializeSVG();
    }

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------

    _initializeSVG() {
        if (typeof d3 === 'undefined') {
            console.error(
                'D3.js is required for GraphRenderer but was not found. ' +
                    'Load d3.v7.min.js before graph.js.'
            );
            return;
        }

        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`GraphRenderer: container #${this.containerId} not found.`);
            return;
        }

        // Fallback dimensions if container has no explicit size
        this.width = container.clientWidth || 800;
        this.height = container.clientHeight || 600;

        this.svg = d3
            .select(`#${this.containerId}`)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height);

        // Zoom + pan support
        const zoomBehavior = d3
            .zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                if (this.zoomGroup) {
                    this.zoomGroup.attr('transform', event.transform);
                }
            });

        this.zoomGroup = this.svg.call(zoomBehavior).append('g');
    }

    // -------------------------------------------------------------------------
    // Rendering
    // -------------------------------------------------------------------------

    /**
     * Render a new graph.
     * @param {Object} graph
     *    graph.nodes: [{ id: string, ... }, ...]
     *    graph.edges: [{ source: string, target: string, ... }, ...]
     * @param {Object|null} scoreMap (optional)
     *    Mapping nodeId -> numeric score (e.g., PageRank or similarity).
     *    Used to scale node radius and color.
     */
    renderGraph(graph, scoreMap = null) {
        if (!this.svg || !this.zoomGroup) {
            console.warn('GraphRenderer: SVG not initialized, aborting renderGraph.');
            return;
        }

        const scores = scoreMap || {};
        this.nodes = (graph.nodes || []).map((node) => ({
            ...node,
            score: typeof scores[node.id] === 'number' ? scores[node.id] : null
        }));

        this.links = (graph.edges || []).map((edge) => ({
            source: edge.source,
            target: edge.target
        }));

        this._updateGraph();
    }

    _updateGraph() {
        const g = this.zoomGroup;
        if (!g) return;

        // Clear previous content
        g.selectAll('*').remove();

        // Basic force simulation
        this.simulation = d3
            .forceSimulation(this.nodes)
            .force(
                'link',
                d3
                    .forceLink(this.links)
                    .id((d) => d.id)
                    .distance(60)
            )
            .force('charge', d3.forceManyBody().strength(-140))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(30));

        // Links (edges)
        const link = g
            .append('g')
            .attr('stroke', '#9ca3af')
            .attr('stroke-opacity', 0.7)
            .selectAll('line')
            .data(this.links)
            .enter()
            .append('line')
            .attr('class', 'graph-link')
            .attr('stroke-width', 1.2);

        // Nodes
        const node = g
            .append('g')
            .selectAll('circle')
            .data(this.nodes)
            .enter()
            .append('circle')
            .attr('class', 'graph-node')
            .attr('r', (d) => this._computeRadius(d.score))
            .attr('fill', (d) => this._computeColor(d.score))
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 1.6)
            .call(
                d3
                    .drag()
                    .on('start', (event, d) => this._onDragStart(event, d))
                    .on('drag', (event, d) => this._onDrag(event, d))
                    .on('end', (event, d) => this._onDragEnd(event, d))
            )
            .on('click', (event, d) => this._onNodeClick(event, d));

        // Node labels
        const label = g
            .append('g')
            .selectAll('text')
            .data(this.nodes)
            .enter()
            .append('text')
            .attr('class', 'graph-label')
            .text((d) => d.label || d.id)
            .attr('font-size', '10px')
            .attr('fill', '#374151')
            .attr('dx', 10)
            .attr('dy', 4)
            .attr('pointer-events', 'none');

        // Tooltips (native browser)
        node.append('title').text((d) => {
            const scoreText =
                typeof d.score === 'number' ? `\nScore: ${d.score.toFixed(4)}` : '';
            return `Node: ${d.id}${scoreText}`;
        });

        // Tick handler for simulation
        this.simulation.on('tick', () => {
            link
                .attr('x1', (d) => d.source.x)
                .attr('y1', (d) => d.source.y)
                .attr('x2', (d) => d.target.x)
                .attr('y2', (d) => d.target.y);

            node.attr('cx', (d) => d.x).attr('cy', (d) => d.y);

            label.attr('x', (d) => d.x).attr('y', (d) => d.y);
        });
    }

    // -------------------------------------------------------------------------
    // Node styling helpers
    // -------------------------------------------------------------------------

    _computeRadius(score) {
        if (typeof score !== 'number') {
            return this.options.defaultRadius;
        }
        const s = Math.max(0, Math.min(1, score)); // clamp to [0,1] if normalized
        const r =
            this.options.minRadius +
            s * (this.options.maxRadius - this.options.minRadius);
        return r;
    }

    _computeColor(score) {
        if (typeof score !== 'number') {
            return '#d1d5db'; // gray
        }
        // Blue (low) -> Purple (mid) -> Red (high)
        const s = Math.max(0, Math.min(1, score));
        const r = Math.floor(80 + 150 * s); // 80-230
        const g = Math.floor(120 + 40 * (1 - s)); // 160-120
        const b = Math.floor(220 - 120 * s); // 220-100
        return `rgb(${r}, ${g}, ${b})`;
    }

    // -------------------------------------------------------------------------
    // Interaction handlers
    // -------------------------------------------------------------------------

    _onDragStart(event, d) {
        if (!event.active && this.simulation) {
            this.simulation.alphaTarget(0.3).restart();
        }
        d.fx = d.x;
        d.fy = d.y;
    }

    _onDrag(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    _onDragEnd(event, d) {
        if (!event.active && this.simulation) {
            this.simulation.alphaTarget(0);
        }
        d.fx = null;
        d.fy = null;
    }

    _onNodeClick(event, d) {
        event.stopPropagation();
        this.highlightNode(d.id);

        // If the global app (from app.js) exposes selectNode, call it.
        if (typeof window !== 'undefined' && window.app && typeof window.app.selectNode === 'function') {
            try {
                window.app.selectNode(d.id);
            } catch (err) {
                console.error('Error calling app.selectNode:', err);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Public helper: highlight a node by id
    // -------------------------------------------------------------------------

    /**
     * Visually highlight a node by id (e.g., when selected in another panel).
     * @param {string} nodeId
     */
    highlightNode(nodeId) {
        if (!this.svg) return;

        this.svg
            .selectAll('.graph-node')
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 1.6)
            .classed('graph-node-selected', false);

        this.svg
            .selectAll('.graph-node')
            .filter((d) => d && d.id === nodeId)
            .attr('stroke', '#f97316')
            .attr('stroke-width', 3)
            .classed('graph-node-selected', true);

        this.selectedNodeId = nodeId;
    }
}

// Optional: auto-create a renderer if there is a #graph container.
// You can comment this out if you prefer manual instantiation.
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('graph');
    if (container && typeof d3 !== 'undefined') {
        const renderer = new GraphRenderer('graph');
        window.graphRenderer = renderer;
    }
});
