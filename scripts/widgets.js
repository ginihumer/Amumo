// define widgets
class ModalityGapWidget {
    constructor(title, data, width = 500, height = 400) {
        this.div = document.createElement('div');
        var trace1 = {
            name: '',
            x: data.distances,
            y: data.losses,
            type: 'scatter',
            mode: 'lines+markers',
            hovertemplate: 'Distance: %{x:.3f} <br>Loss: %{y:.3f}'
        };

        var layout = {
            xaxis: {
                title: 'Euclidean Distance',
                zeroline: false,
            },
            yaxis: {
                title: 'Loss',
            },
            width: width,
            height: height,
            title: title,
            shapes: [
                {
                    type: 'line',
                    x0: data.original_distance,
                    x1: data.original_distance,
                    y0: 0,
                    y1: Math.max(...data.losses) * 1.2,
                    line: {
                        width: 1,
                        dash: 'dash'
                    }
                }
            ]
        }

        Plotly.newPlot(this.div, [trace1], layout, { displayModeBar: false });
    }
}

class HoverWidget {
    constructor(dataset_name, text_list, width = 400, height = 200) {
        this.dataset_name = dataset_name;
        this.text_list = text_list;
        this.width = width;
        this.height = height;

        this.audio_muted = true;

        this.div1 = document.createElement('div');
        this.div1.style.wordWrap = 'break-word';
        this.div1.style.minWidth = '200px';
        this.div1.style.marginBottom = '20px';
        this.div2 = document.createElement('div');
        this.div2.style.wordWrap = 'break-word';
        this.div2.style.minWidth = '200px';
        this.div3 = document.createElement('div');
        this.div3.style.wordWrap = 'break-word';
        this.div3.style.minWidth = '200px';

        this.div = document.createElement('div');
        // this.div.style.width = width + 'px';
        // this.div.style.height = height + 'px';
        this.div.appendChild(this.div1);
        this.div.appendChild(this.div2);
        this.div.appendChild(this.div3);
        this.div.style.maxHeight = height + 'px';
    }

    update1(type, idx) {
        this.update(this.div1, type, idx);
    }
    update2(type, idx) {
        this.update(this.div2, type, idx);
    }
    update3(type, idx) {
        this.update(this.div3, type, idx);
    }

    update(div, type, idx) {
        if (type == 'Image') {
            const img = document.createElement('img');
            img.src = './exported_data/' + this.dataset_name + '/images/' + idx + '.jpg';
            // img.width = this.width;
            // img.height = this.height;
            img.style.maxWidth = this.width / 2 + 'px';
            img.style.maxHeight = this.height + 'px';
            div.innerHTML = '';
            div.appendChild(img);
        } else if (type == 'Text') {
            div.innerHTML = this.text_list[idx];
        } else if (type == 'Audio') {
            const wav = document.createElement('audio');
            wav.controls = true;
            wav.autoplay = true;
            wav.muted = this.audio_muted;
            wav.src = './exported_data/' + this.dataset_name + '/audios/' + idx + '.wav';
            wav.onvolumechange = () => {
                this.audio_muted = wav.muted;
            }
            div.innerHTML = '';
            div.appendChild(wav);
        } else {
            div.innerHTML = '';
        }
    }
}


class ScatterPlotWidget {
    projection_methods = ['PCA', 'TSNE', 'UMAP'];

    constructor(projection_df, model_name, selected_method = 'PCA', title = '', width = 400, height = 300, modalities = {"Image": "Image", "Text": "Text"}) {
        this.modalities = modalities;
        this.nr_modalities = Object.keys(modalities).length;
        this.size = 100;

        let margin_top = 30;
        if (title == '') {
            margin_top = 10;
        }
        this.projection_df = projection_df;
        this.model_name = model_name;
        this.width = width;
        this.height = height;

        // projection method selection
        this.method_select = document.createElement('select');
        for (const key in this.projection_methods) {
            var opt = new Option(this.projection_methods[key], this.projection_methods[key])
            this.method_select.appendChild(opt);
        }
        this.method_select.selectedIndex = this.projection_methods.findIndex((value) => value == selected_method);
        this.method_select.onchange = () => {
            this.update_scatter();
        }


        // scatter plot
        this.scatter_div = document.createElement('div');

        const COLORS = ["blue", "orange", "green"];
        let traces = [];
        for(let i = 0; i < this.nr_modalities; i++){
            traces.push({
                name: Object.values(this.modalities)[i],
                x: Array(this.size).fill(0),
                y: Array(this.size).fill(0),
                type: 'scatter',
                mode: 'markers',
                hoverinfo: 'text',
                marker: {
                    color: COLORS[i]
                }
            });
        }
        // traces.push({
        //     name: "connections",
        //     x: [0,0],
        //     y: [1,1],
        //     type: 'line',
        //     mode: 'lines',
        //     hoverinfo: 'text',
        //     marker: {
        //         color: 'grey'
        //     }
        // })


        // var trace_highlight = {
        //     name: 'hover',
        //     x: [],
        //     y: [],
        //     type: 'scatter',
        //     mode: 'markers',
        //     hoverinfo: 'text',
        //     marker: {
        //         size: 10,
        //         color: 'black'
        //     }
        // };
        // traces.push(trace_highlight);

        const layout = {
            width: this.width,
            height: this.height,
            margin: { l: 10, r: 10, t: margin_top, b: 10 },
            legend: {
                yanchor: 'top',
                y: 0.99,
                xanchor: 'left',
                x: 0.01
            },
            xaxis: {
                tickmode: 'none',
                showticklabels: false,
                // showgrid: false,
                zeroline: false,
            },
            yaxis: {
                tickmode: 'none',
                showticklabels: false,
                // showgrid: false,
                zeroline: false,
            },
            title: title,
        }
        Plotly.newPlot(this.scatter_div, traces, layout, { modeBarButtonsToRemove: ['select2d', 'lasso2d'] }); //'pan2d','select2d','lasso2d','resetScale2d','zoomOut2d'

        this.update_scatter();

        this.div = document.createElement('div');
        this.div.appendChild(this.method_select);
        this.div.appendChild(this.scatter_div);

        const this_ = this;
        const evnt = function(data){
            this_.highlight('reset', 0);
            const idx = data.points[0].pointIndex;
            for (let modality in this_.modalities) {
                this_.highlight(modality, idx);
            }
            this_.highlight('connection_line', idx);
            this_.highlight('update', 0);
        };
        this.scatter_div.on('plotly_hover', evnt);
        this.scatter_div.on('plotly_click', evnt);
        this.scatter_div.on('plotly_unhover', function(data){
            this_.highlight('reset', 0);
            this_.highlight('update', 0);
        });
    }

    get_XY_coordinates() {
        let coordinates_x = Array(this.nr_modalities);
        let coordinates_y = Array(this.nr_modalities);
        for (let i = 0; i < this.nr_modalities; i++) {
            const modality = Object.keys(this.modalities)[i];
            const x = this.projection_df.filter((row) => row.data_type == modality.toLowerCase()).map(row => +row[this.model_name + '_' + this.method_select.value + '_x'])
            const y = this.projection_df.filter((row) => row.data_type == modality.toLowerCase()).map(row => +row[this.model_name + '_' + this.method_select.value + '_y'])
            coordinates_x[i] = x;
            coordinates_y[i] = y;
        }
        return {"x": coordinates_x, "y": coordinates_y};
    }

    update_scatter() {
        const coords = this.get_XY_coordinates();
        const coords_x_T = coords.x[0].map((_, colIndex) => coords.x.map(row => row[colIndex]))
        const coords_y_T = coords.y[0].map((_, colIndex) => coords.y.map(row => row[colIndex]))
        
        let lines = []
        for (let line_idx = 0; line_idx < this.size; line_idx++) {
            const curr_line_x = coords_x_T[line_idx];
            const centroid_x = curr_line_x.reduce((prev, curr) => prev + curr, 0) / curr_line_x.length;
            const curr_line_y = coords_y_T[line_idx];
            const centroid_y = curr_line_y.reduce((prev, curr) => prev + curr, 0) / curr_line_y.length;
            
            for (let modality_idx = 0; modality_idx < this.nr_modalities; modality_idx++) {
                const line = {
                    name: 'connections',
                    type: 'line',
                    x0: centroid_x,
                    y0: centroid_y,
                    x1: curr_line_x[modality_idx],
                    y1: curr_line_y[modality_idx],
                    line: {
                        width: 1,
                        color: 'grey'
                    },
                    opacity:0.4
                }
                lines.push(line);
            }
        }
        // coords.x.push([0,0]), coords.y.push([1,1]); 
        Plotly.update(this.scatter_div, { 'x': coords.x, 'y': coords.y }, { shapes: lines });
    }

    highlight_shapes_buffer = [];
    highlight(mode, idx){
        // use mode "update" to trigger the update of the highlight shapes
        if(mode == 'update'){
            // TODO: updating takes too much time; the lag stems from redrawing the grey connection lines between the pairs
            // Plotly.update(this.scatter_div, {}, {shapes: this.highlight_shapes_buffer});
            return;
        }
        // let shapes = this.scatter_div.layout.shapes;
        if(mode == 'reset'){
            this.highlight_shapes_buffer = this.scatter_div.layout.shapes?.filter((shape) => shape.name !== 'hover_shape');
            // Plotly.update(this.scatter_div, {}, {shapes: shapes});
            return;
        }

        const coords = this.get_XY_coordinates();
        if(mode == 'connection_line'){
            const coords_x_T = coords.x[0].map((_, colIndex) => coords.x.map(row => row[colIndex]))
            const coords_y_T = coords.y[0].map((_, colIndex) => coords.y.map(row => row[colIndex]))
            
            const curr_line_x = coords_x_T[idx];
            const centroid_x = curr_line_x.reduce((prev, curr) => prev + curr, 0) / curr_line_x.length;
            const curr_line_y = coords_y_T[idx];
            const centroid_y = curr_line_y.reduce((prev, curr) => prev + curr, 0) / curr_line_y.length;
            
            for (let modality_idx = 0; modality_idx < this.nr_modalities; modality_idx++) {
                const line = {
                    name: 'hover_shape',
                    type: 'line',
                    x0: centroid_x,
                    y0: centroid_y,
                    x1: curr_line_x[modality_idx],
                    y1: curr_line_y[modality_idx],
                    line: {
                        width: 1,
                        color: 'yellow'
                    },
                }
                this.highlight_shapes_buffer.push(line);
            }
            // Plotly.update(this.scatter_div, {}, {shapes: shapes});
            return;
        }

        const modalities = Object.keys(this.modalities)
        
        const x = coords.x[modalities.indexOf(mode)][idx]
        const y = coords.y[modalities.indexOf(mode)][idx]

        const range_x = this.scatter_div.layout.xaxis.range[1] - this.scatter_div.layout.xaxis.range[0]
        const range_y = this.scatter_div.layout.yaxis.range[1] - this.scatter_div.layout.yaxis.range[0]
        const r = [range_x*0.01, range_y*0.01]
        
        let highlight_points = [{
            name: 'hover_shape',
            type: 'circle',
            x0: x-r[0],
            x1: x+r[0],
            y0: y-r[1],
            y1: y+r[1],
            line: {
                width: 2,
                color: 'yellow'
            },
            fillcolor: "yellow"
        },]
        this.highlight_shapes_buffer.push(...highlight_points);
        // Plotly.update(this.scatter_div, {}, {shapes: shapes});
    }
}


class SimilarityHeatmapWidget {
    constructor(do_cluster = false, title = '', width = 500, height = 420, z_min = null, z_max = null, size = 100, cluster_between = ["Image", "Text"], modalities = {"Image": "Image", "Text": "Text"}) {
        this.do_cluster = do_cluster;
        this.cluster_between = cluster_between;
        this.size = size;
        this.n_modalities = Object.keys(modalities).length;
        this.modalities = modalities;

        const data = Array.from(Array(size * this.n_modalities), () => Array(size * this.n_modalities).fill(0));
        var trace1 = {
            name: '',
            z: data,
            type: 'heatmap',
            // hoverinfo: 'text',
            hovertemplate: '%{z:.3f}',
            colorscale: plasma_colors, //'Viridis',//'YlOrRd', 
            colorbar: {"title": 'Similarity'},
            // reversescale: true
            zmin: z_min,
            zmax: z_max,
        };
        var traces = [trace1];

        var layout = {
            width: width,
            height: height,
            margin: { l: 65, r: 10, t: 10, b: 25 },
        }
        this.heatmap_div = document.createElement('div');
        Plotly.newPlot(this.heatmap_div, traces, layout);

        this.meta_info_div = document.createElement('div');
        this.meta_info_div.style.fontSize = '12px';
        this.meta_info_div.style.marginLeft = '50px';

        this.div = document.createElement('div');
        if (title !== '') {
            const title_el = document.createElement('h4');
            title_el.innerHTML = title;
            title_el.style.marginLeft = '50px';
            title_el.style.marginBottom = '0px';
            this.div.appendChild(title_el);
        }
        this.div.appendChild(this.meta_info_div);
        this.div.appendChild(this.heatmap_div);

    }

    _set_plotly_event() {
        const this_ = this;
        const evnt = function (data) {
            const idx = data.points[0].pointIndex;
            this_.update_hoverIdx([idx]);
        };
        this.heatmap_div.on('plotly_hover', evnt)
        this.heatmap_div.on('plotly_click', evnt)
    }

    _get_matrix_gridlines(data) {
        let gridlines = []
        for(let i = 1; i < this.n_modalities; i++){
            gridlines.push(
                {
                    type: 'line',
                    x0: i * data.length / this.n_modalities - 0.5,
                    x1: i * data.length / this.n_modalities - 0.5,
                    y0: 0 - 0.5,
                    y1: data.length - 0.5,
                    line: {
                        width: 1,
                        color: 'black'
                    }
                },
                {
                    type: 'line',
                    y0: i * data.length / this.n_modalities - 0.5,
                    y1: i * data.length / this.n_modalities - 0.5,
                    x0: 0 - 0.5,
                    x1: data.length - 0.5,
                    line: {
                        width: 1,
                        color: 'black'
                    }
                },)
        }
        return gridlines
    }

    async update_heatmap(dataset_name, model_name, show_meta_info = true) {
        const _this = this;
        return load_meta_info_fn(dataset_name, model_name).then(async (meta_data) => {

            if (show_meta_info) {
                if(typeof meta_data.gap_distance === 'number'){
                    this.meta_info_div.innerHTML = `Modality distance: ${meta_data.gap_distance.toFixed(2)} | Loss: ${meta_data.loss.toFixed(2)}`;
                }else if(typeof meta_data.gap_distance === 'object'){
                    const modality_gaps = Object.keys(meta_data.gap_distance).map((key) => { 
                        return key.split("_").join(' & ') + ": " + meta_data.gap_distance[key].toFixed(2)
                    }).join(' | ')
                    const losses = Object.keys(meta_data.loss).map((key) => { 
                        return key.split("_").join(' & ') + ": " + meta_data.loss[key].toFixed(2)
                    }).join(' | ')
                    this.meta_info_div.innerHTML = `Modality distance: ${modality_gaps} <br> Loss: ${losses}`;
                }
            }

            var cluster_info = meta_data; // this way of handing clusters is deprecated; use the "clusters" dict for giving clusters for each mode pair

            if (Object.keys(meta_data).includes("clusters")){
                cluster_info = meta_data["clusters"][this.cluster_between[0].toLowerCase() + '_' + this.cluster_between[1].toLowerCase()];
            }
            this.cluster_sort_idcs = cluster_info.cluster_sort_idcs;
            this.cluster_sort_idcs_reverse = cluster_info.cluster_sort_idcs_reverse;
            
            await load_similarities_fn(dataset_name, model_name).then((data) => {
                if (this.do_cluster) {
                    var clustered_data = Array.from(Array(data.length), () => Array(data[0].length).fill(0));

                    for (let i = 0; i < this.cluster_sort_idcs.length; i++) {
                        const i_x = this.cluster_sort_idcs[i];
                        for (let j = 0; j < this.cluster_sort_idcs.length; j++) {
                            const i_y = this.cluster_sort_idcs[j];

                            // sort and add to clustered array for each quadrant
                            for (let m_x = 0; m_x < this.n_modalities; m_x++) {
                                for (let m_y = 0; m_y < this.n_modalities; m_y++) {
                                    clustered_data[i + m_x * this.size][j + m_y * this.size] = data[i_x + m_x * this.size][i_y + m_y * this.size];
                                    // clustered_data[i][j + this.cluster_sort_idcs.length] = data[i_x][i_y + this.cluster_sort_idcs.length];
                                    // clustered_data[i + this.cluster_sort_idcs.length][j] = data[i_x + this.cluster_sort_idcs.length][i_y];
                                    // clustered_data[i + this.cluster_sort_idcs.length][j + this.cluster_sort_idcs.length] = data[i_x + this.cluster_sort_idcs.length][i_y + this.cluster_sort_idcs.length];
                                }
                            }
                        }
                    }
                    data = clustered_data;
                }

                const shapes = this._get_matrix_gridlines(data);

                if (this.do_cluster) {
                    const cluster_labels = cluster_info.cluster_labels;
                    const cluster_sizes = cluster_info.cluster_sizes;

                    const modalities = Object.keys(_this.modalities)

                    let offset = 0 - 0.5 // -0.5 because heatmap rectangles are drawn around [-0.5, 0.5]
                    for (let i = 0; i < cluster_labels.length; i++) {
                        const cluster_label = cluster_labels[i];
                        const cluster_size = cluster_sizes[i];

                        if (cluster_size > 5) {
                            let textposition = 'middle right';
                            if (offset < _this.size / 2) {
                                textposition = 'middle left';
                            }
                            shapes.push(
                                {
                                    type: 'rect',
                                    x0: modalities.indexOf(cluster_between[1])*_this.size + offset,
                                    y0: modalities.indexOf(cluster_between[0])*_this.size + offset,
                                    x1: modalities.indexOf(cluster_between[1])*_this.size + offset + cluster_size,
                                    y1: modalities.indexOf(cluster_between[0])*_this.size + offset + cluster_size,
                                    label: {
                                        text: cluster_label,
                                        textposition: textposition,
                                        font: { size: 10, color: 'white' },
                                        padding: Math.log(cluster_size) * 10
                                    },
                                    line: {
                                        width: 1,
                                        color: 'white'
                                    },
                                }
                            );
                        }
                        offset += cluster_size;
                    }

                }

                let tickvals = [];
                for(let i = 0; i < this.n_modalities; i++){
                    tickvals.push(i * this.size + this.size / 2);
                }

                Plotly.update(this.heatmap_div, { 'z': [data] }, {
                    xaxis: {
                        tickmode: 'array',
                        ticktext: Object.values(this.modalities),
                        tickvals: tickvals,
                        fixedrange: false
                    },
                    yaxis: {
                        tickmode: 'array',
                        tickvals: tickvals,
                        ticktext: Object.values(this.modalities),
                        fixedrange: false,
                        autorange: 'reversed',
                    },
                    shapes: shapes
                }, [0]);

                this._set_plotly_event();

            });
        });

    }

    update_hoverIdx(idcs) {
        const data = this.heatmap_div.data[0].z;
        let shapes = this.heatmap_div.layout.shapes?.filter((shape) => shape.name !== 'hover_idx');
        for (let i = 0; i < idcs.length; i++) {
            const y_idx = idcs[i][0];
            const x_idx = idcs[i][1];

            if (x_idx >= 0 & x_idx < data.length) {
                const shape = {
                    name: 'hover_idx',
                    type: 'line',
                    x0: x_idx,
                    x1: x_idx,
                    y0: 0 - 0.5,
                    y1: data.length - 0.5,
                    line: {
                        width: 1,
                        color: 'grey'
                    }
                }
                shapes.push(shape)
            }
            if (y_idx >= 0 & y_idx < data.length) {
                const shape = {
                    name: 'hover_idx',
                    type: 'line',
                    y0: y_idx,
                    y1: y_idx,
                    x0: 0 - 0.5,
                    x1: data.length - 0.5,
                    line: {
                        width: 1,
                        color: 'grey'
                    }
                }
                shapes.push(shape)
            }
        }
        Plotly.update(this.heatmap_div, {}, { shapes: shapes });
    }

}


// helper functions

function connect_scatter_hover(scatter_widget, hover_widget) {
    const evnt = function (data) {
        const idx = data.points[0].pointIndex;

        if(Object.keys(scatter_widget.modalities).includes('Text'))
            hover_widget.update1('Text', idx);
        else
            hover_widget.update1('', 0);

        if(Object.keys(scatter_widget.modalities).includes('Audio'))
            hover_widget.update2('Audio', idx);
        else
            hover_widget.update2('', 0);

        if(Object.keys(scatter_widget.modalities).includes('Image'))
            hover_widget.update3('Image', idx);
        else
            hover_widget.update3('', 0);
    };

    scatter_widget.scatter_div.on('plotly_hover', evnt)
    scatter_widget.scatter_div.on('plotly_click', evnt)
}

function connect_heatmap_hover(heatmap_widget, hover_widget) {
    const evnt = function (data) {
        const idx = data.points[0].pointIndex;
        let y_idx = idx[0];
        let x_idx = idx[1];

        mode_x_id = Math.floor(x_idx / heatmap_widget.size);
        mode_y_id = Math.floor(y_idx / heatmap_widget.size);

        mode_x = Object.keys(heatmap_widget.modalities)[mode_x_id];
        mode_y = Object.keys(heatmap_widget.modalities)[mode_y_id];

        x_idx = x_idx % (heatmap_widget.size);
        y_idx = y_idx % (heatmap_widget.size);

        if (heatmap_widget.do_cluster) {
            y_idx = heatmap_widget.cluster_sort_idcs[y_idx];
            x_idx = heatmap_widget.cluster_sort_idcs[x_idx];
        }

        hover_widget.update1(mode_x, x_idx);
        hover_widget.update2(mode_y, y_idx);
        hover_widget.update3('', 0);
    };
    heatmap_widget.heatmap_div.on('plotly_hover', evnt)
    heatmap_widget.heatmap_div.on('plotly_click', evnt)
}

function connect_scatter_heatmap(scatter_widget, heatmap_widget) {
    const scatter_evnt = function (data) {
        heatmap_widget.update_hoverIdx([])
        const idx = data.points[0].pointIndex;
        heatmap_idx = idx;
        if (heatmap_widget.do_cluster) {
            heatmap_idx = heatmap_widget.cluster_sort_idcs_reverse[idx];
        }
        let hover_idcs = []
        for(let i = 0; i < scatter_widget.nr_modalities; i++){
            hover_idcs.push([scatter_widget.size * i + heatmap_idx, scatter_widget.size * i + heatmap_idx]);
        }
        heatmap_widget.update_hoverIdx(hover_idcs)
    }
    scatter_widget.scatter_div.on('plotly_hover', scatter_evnt);
    scatter_widget.scatter_div.on('plotly_click', scatter_evnt);

    scatter_widget.scatter_div.on('plotly_unhover', function (data) {
        heatmap_widget.update_hoverIdx([])
    });

    const heatmap_evnt = function(data){
        scatter_widget.highlight('reset', 0);

        const idx = data.points[0].pointIndex;
        let y_idx = idx[0];
        let x_idx = idx[1];
        
        mode_x_id = Math.floor(x_idx / heatmap_widget.size);
        mode_y_id = Math.floor(y_idx / heatmap_widget.size);

        mode_x = Object.keys(heatmap_widget.modalities)[mode_x_id];
        mode_y = Object.keys(heatmap_widget.modalities)[mode_y_id];

        x_idx = x_idx % (heatmap_widget.size);
        y_idx = y_idx % (heatmap_widget.size);

        if(heatmap_widget.do_cluster){
          y_idx = heatmap_widget.cluster_sort_idcs[y_idx];
          x_idx = heatmap_widget.cluster_sort_idcs[x_idx];
        }

        scatter_widget.highlight(mode_x, x_idx);
        scatter_widget.highlight(mode_y, y_idx);
        if (mode_x !== mode_y && x_idx === y_idx){
            scatter_widget.highlight('connection_line', y_idx);
        }
        scatter_widget.highlight('update', 0);
    };
    heatmap_widget.heatmap_div.on('plotly_hover', heatmap_evnt);
    heatmap_widget.heatmap_div.on('plotly_click', heatmap_evnt);

    heatmap_widget.heatmap_div.on('plotly_unhover', function(data){
        scatter_widget.highlight('reset', 0);
        scatter_widget.highlight('update', 0);
    });
}

function clip_explorer_by_model(dataset_name, model_name, el, prompts_promise, projection_promise, projection_method = 'PCA', do_cluster = false, cluster_between = ["Image", "Text"], modalities = {"Image": "Image", "Text": "Text"}) {

    if (!(el instanceof Element)) {
        el = document.getElementById(el);
    }
    projection_promise.then(function (data) {
        const heatmap_widget = new SimilarityHeatmapWidget(do_cluster = do_cluster, title = '', width = 500, height = 420, z_min = null, z_max = null, size = 100, cluster_between = cluster_between, modalities = modalities);
        heatmap_widget.update_heatmap(dataset_name, model_name).then(() => {
            heatmap_widget.div.classList.add('col-xl-5', 'col-sm-auto');
            el.appendChild(heatmap_widget.div)

            const scatter_widget = new ScatterPlotWidget(data, model_name, selected_method = projection_method, title = '', width = 400, height = 300, modalities = modalities);
            scatter_widget.div.classList.add('col-xl-4', 'col-sm-auto');
            el.appendChild(scatter_widget.div)

            connect_scatter_heatmap(scatter_widget, heatmap_widget);

            prompts_promise.then(captions => {
                const hover_widget = new HoverWidget(dataset_name, captions);
                connect_scatter_hover(scatter_widget, hover_widget)
                connect_heatmap_hover(heatmap_widget, hover_widget);
                hover_widget.div.classList.add('col-xl-3', 'col-sm-auto');
                el.appendChild(hover_widget.div)
                return captions;
            });
        });
        return data;
    });
}


const AVAILABLE_MODELS = ['CLIP', 'CyCLIP', 'CLOOB', 'CLOOB_LAION400M'];
function clip_explorer_widget(dataset_name, el_id, prompts_promise, projection_promise, projection_method = 'UMAP', available_models = AVAILABLE_MODELS, modalities = {"Image": "Image", "Text": "Text"}) {
    const div = document.getElementById(el_id);

    // set modality values if null -> keys are used as ID's for data selection and filtering; values are used for display
    for(const key in modalities){
        if(modalities[key] === null){
            modalities[key] = key;
        }
    }

    // init model select
    const model_select = document.createElement('select');
    for (const key in available_models) {
        var opt = new Option(available_models[key], available_models[key])
        model_select.appendChild(opt);
    }
    model_select.selectedIndex = 0;
    model_select.style.marginLeft = '50px';
    if (available_models.length <= 1) {
        model_select.style.display = 'none';
    }
    div.appendChild(model_select);

    // init modality gap checkbox
    const close_gap = document.createElement('input');
    close_gap.type = 'checkbox';
    close_gap.id = el_id + '_close_gap';
    close_gap.style = 'margin-left: 20px;';
    const label = document.createElement('label');
    label.setAttribute('for', close_gap.id);
    label.textContent = 'Close modality gap';
    label.style = 'font-size:15px; margin-left:5px;';
    if(Object.keys(modalities).length > 2){
        close_gap.style.display = 'none';
        label.style.display = 'none';
    }
    div.appendChild(close_gap);
    div.appendChild(label);

    // init cluster checkbox
    const cluster_checkbox = document.createElement('input');
    cluster_checkbox.type = 'checkbox';
    cluster_checkbox.id = el_id + '_cluster'
    cluster_checkbox.style = 'margin-left: 20px;'
    const cluster_label = document.createElement('label');
    cluster_label.setAttribute('for', cluster_checkbox.id);
    cluster_label.textContent = 'Cluster matrix by similarity';
    cluster_label.style = 'font-size:15px; margin-left:5px;';
    div.appendChild(cluster_checkbox);
    div.appendChild(cluster_label);

    // cluster similarities between modalities select
    const cluster_between_1_select = document.createElement('select');
    cluster_between_1_select.id = el_id + '_cluster_between_1';
    cluster_between_1_select.style = 'min-width: 100px;';
    cluster_between_1_select.disabled = ! cluster_checkbox.checked;
    const cluster_between_1_label = document.createElement('label');
    cluster_between_1_label.setAttribute('for', cluster_between_1_select.id);
    cluster_between_1_label.textContent = 'between';
    cluster_between_1_label.style = 'font-size:15px; margin-left:5px; margin-right:5px;';
    
    const cluster_between_2_select = document.createElement('select');
    cluster_between_2_select.id = el_id + '_cluster_between_2'
    cluster_between_2_select.style = 'min-width: 100px;';
    cluster_between_2_select.disabled = !cluster_checkbox.checked;
    const cluster_between_2_label = document.createElement('label');
    cluster_between_2_label.setAttribute('for', cluster_between_2_select.id);
    cluster_between_2_label.textContent = 'and';
    cluster_between_2_label.style = 'font-size:15px; margin-left:5px; margin-right:5px;';

    for (const key in modalities) {
        var opt = new Option(key, key)
        cluster_between_1_select.appendChild(opt);
        var opt = new Option(key, key)
        cluster_between_2_select.appendChild(opt);
    }
    cluster_between_1_select.selectedIndex = 0;
    cluster_between_2_select.selectedIndex = 1;

    if(Object.keys(modalities).length <= 2){
        cluster_between_1_select.style.display = 'none';
        cluster_between_1_label.style.display = 'none';
        cluster_between_2_select.style.display = 'none';
        cluster_between_2_label.style.display = 'none';
    }
    
    div.appendChild(cluster_between_1_label);
    div.appendChild(cluster_between_1_select);
    div.appendChild(cluster_between_2_label);
    div.appendChild(cluster_between_2_select);
    
    
    // init explorer div
    const explorer_div = document.createElement('div');
    explorer_div.style = 'display: flex;'
    explorer_div.classList.add('row', 'd-flex', 'justify-content-center');
    div.appendChild(explorer_div);

    // handle UI changes
    const update_explorer = () => {
        explorer_div.innerHTML = ''
        let model_name = model_select.value;
        if (close_gap.checked) {
            model_name += '_nogap';
        }
        clip_explorer_by_model(dataset_name, model_name, explorer_div, prompts_promise, projection_promise, projection_method, do_cluster = cluster_checkbox.checked, cluster_between = [cluster_between_1_select.value, cluster_between_2_select.value], modalities = modalities)
    };
    model_select.addEventListener("change", update_explorer);
    close_gap.addEventListener("change", update_explorer);
    cluster_checkbox.addEventListener("change", update_explorer);
    cluster_checkbox.addEventListener("change", () => {
        cluster_between_1_select.disabled = !cluster_checkbox.checked;
        cluster_between_2_select.disabled = !cluster_checkbox.checked;
    });
    cluster_between_1_select.addEventListener("change", update_explorer);
    cluster_between_2_select.addEventListener("change", update_explorer);

    update_explorer();

}

clip_comparer = (models, prompts_promise, dataset_name, el_id, z_min = null, z_max = null, show_meta_info = false) => {
    // models can either be a list of strings or an object of {model_title: model}
    if(models instanceof Array){
        models = models.reduce((a, v) => ({ ...a, [v]: v}), {}) 
    }
    
    prompts_promise.then(async captions => {
        let heatmap_widgets = {}

        function highlight_hover(data) {
            for (const key in heatmap_widgets) {
                if (Object.hasOwnProperty.call(heatmap_widgets, key)) {
                    const element = heatmap_widgets[key];
                    const idx = data.points[0].pointIndex;
                    element.update_hoverIdx([idx]);
                }
            }
        }

        const hover_widget = new HoverWidget(dataset_name, captions);

        for (const model_title in models) {
            const model = models[model_title];
            heatmap_widgets[model_title] = new SimilarityHeatmapWidget(do_cluster = false, title = model_title, width = 500, height = 420, z_min = z_min, z_max = z_max);
            await heatmap_widgets[model_title].update_heatmap(dataset_name, model, show_meta_info = show_meta_info);
            heatmap_widgets[model_title].div.classList.add('col-md-6', 'col-xs-12');
            heatmap_widgets[model_title].div.style.marginTop = '10px';
            document.getElementById(el_id).appendChild(heatmap_widgets[model_title].div)
            heatmap_widgets[model_title].heatmap_div.on('plotly_hover', highlight_hover)
            heatmap_widgets[model_title].heatmap_div.on('plotly_click', highlight_hover)
            connect_heatmap_hover(heatmap_widgets[model_title], hover_widget);
        }

        document.getElementById(el_id + '-hover').appendChild(hover_widget.div)

        return captions;
    })
}


const augmented_heatmap_comparer = (augmentation, el_id) => {

    const img_thumbnails = [];

    const update_augmented_clip_comparer = (id) => {

        img_thumbnails.forEach((img) => {
            img.style.borderColor = 'transparent';
            img.style.opacity = 0.3;
        });
        img_thumbnails[id].style.borderColor = 'black';
        img_thumbnails[id].style.opacity = 0.9;

        const dataset_name = augmentation + '-' + id + '_size-100';

        const prompts_promise = fetch('./exported_data/' + dataset_name + '/prompts.txt')
            .then(response => response.text())
            .then(data => data.split('\n'))
            .catch(error => console.error(error));

        document.getElementById(el_id).innerHTML = '';
        document.getElementById(el_id + '-hover').innerHTML = '';

        clip_comparer(['CLIP', 'CyCLIP', 'CLOOB', 'CLOOB_LAION400M'], prompts_promise, dataset_name, el_id, z_min = 0, z_max = 1)
    }

    document.getElementById(el_id + '-picker').innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const img = document.createElement('img');
        img.src = './exported_data/example_images/' + i + '.jpg';
        img.style.cursor = 'pointer';
        img.style.border = 'transparent 2px solid';
        img.style.width = '100px';
        img.style.height = '100px';
        document.getElementById(el_id + '-picker').appendChild(img);
        img_thumbnails.push(img);
        img.addEventListener("click", function () {
            update_augmented_clip_comparer(i)
        });
    }

    update_augmented_clip_comparer(0);

}