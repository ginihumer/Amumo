# ipywidgets meets plotly: https://plotly.com/python/figurewidget-app/

import traitlets
from ipywidgets import widgets
import plotly.graph_objects as go
import io
from PIL import Image
import numpy as np

import torch

import math

from . import model as am_model
from . import data as am_data
from .utils import get_textual_label_for_cluster, get_embeddings_per_modality, get_embedding, get_similarities, get_similarity, get_similarities_all, get_cluster_sorting, get_modality_distance, calculate_val_loss, get_closed_modality_gap, get_modality_gap_normed, l2_norm, get_gap_direction


class SimilarityHeatmapWidget(widgets.VBox):
    
    value = traitlets.Any(np.zeros((6,6))).tag(sync=True)
    cluster = traitlets.Any().tag(sync=True)

    hover_idx = traitlets.List([]).tag(sync=True)


    def __init__(self, zmin=None, zmax=None):
        super(SimilarityHeatmapWidget, self).__init__()

        self.fig_widget = go.FigureWidget(data=[go.Heatmap(z=self.value, zmin=zmin, zmax=zmax)])
        self.heatmap = self.fig_widget.data[0]
        self.heatmap.hoverinfo = "text"
        self.fig_widget.update_layout(width=500, height=420,
            xaxis = dict(
                tickmode = 'array',
                tickvals = [len(self.value)/4, 3*len(self.value)/4],
                ticktext = ['Image', 'Text']
            ),
            yaxis = dict(
                tickmode = 'array',
                tickvals = [len(self.value)/4, 3*len(self.value)/4],
                ticktext = ['Image', 'Text']
            ),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        self.fig_widget.update_yaxes(autorange='reversed', fixedrange=False)
        self.fig_widget.update_xaxes(fixedrange=False)
        self.fig_widget.layout.shapes = self._get_matrix_gridlines()

        self.children = [widgets.HBox([self.fig_widget])]



    def _get_matrix_gridlines(self):
        return [
            go.layout.Shape(type='line', x0=len(self.value)/2-0.5, y0=0-0.5, x1=len(self.value)/2-0.5, y1=len(self.value)-0.5, line=dict(color="black", width=1)),
            go.layout.Shape(type='line', y0=len(self.value)/2-0.5, x0=0-0.5, y1=len(self.value)/2-0.5, x1=len(self.value)-0.5, line=dict(color="black", width=1))
        ]


    @traitlets.validate("value")
    def _validate_value(self, proposal):
        # print("TODO: validate value")
        return proposal.value

    @traitlets.observe("value")
    def onUpdateValue(self, change):
        self.fig_widget.data[0].z = self.value
        self.fig_widget.layout.shapes = self._get_matrix_gridlines()

        self.fig_widget.update_layout(
            xaxis = dict(tickvals = [len(self.value)/4, 3*len(self.value)/4]),
            yaxis = dict(tickvals = [len(self.value)/4, 3*len(self.value)/4])
        )


    # @traitlets.validate("cluster")
    # def _validate_cluster(self, proposal):
        # takes a list of cluster labels + sizes
    #     print("TODO: validate cluster")
    #     return proposal.value

    @traitlets.observe("cluster")
    def onUpdateCluster(self, change):
        cluster_shapes = self._get_matrix_gridlines()
        labels, sizes = self.cluster
        offset = 0-0.5 # -0.5 because heatmap rectangles are drawn around [-0.5, 0.5]
        for (cluster_label, cluster_size) in zip(labels, sizes):
            if cluster_size > 5:
                textposition = 'middle left' if offset < len(self.value)/2/2 else 'middle right'

                # see https://plotly.com/python/shapes/
                cluster_shapes += [go.layout.Shape(
                    type='rect', 
                    x0=len(self.value)/2+offset, 
                    y0=offset, 
                    x1=len(self.value)/2+offset+cluster_size, 
                    y1=offset+cluster_size, 
                    label=dict(text=cluster_label, textposition=textposition, font=dict(size=10, color="white"), padding=np.log(cluster_size)*10), 
                    line=dict(width=1, color='white')
                )]

            offset += cluster_size
            
        self.fig_widget.layout.shapes = cluster_shapes


    @traitlets.observe("hover_idx")
    def onUpdateHoverIdx(self, change):
        shapes = [sh for sh in self.fig_widget.layout.shapes if sh.name != 'hover_idx' and sh.name != 'hover_idx']

        for (x_idx, y_idx) in self.hover_idx:
            if x_idx >= 0 and x_idx < len(self.value):
                shapes.append(go.layout.Shape(name='hover_idx', type='line', x0=x_idx, y0=0-0.5, x1=x_idx, y1=len(self.value)-0.5, line=dict(color="grey", width=1)))
            if y_idx >= 0 and y_idx < len(self.value):
                shapes.append(go.layout.Shape(name='hover_idx', type='line', y0=y_idx, x0=0-0.5, y1=y_idx, x1=len(self.value)-0.5, line=dict(color="grey", width=1)))
        
        self.fig_widget.layout.shapes = shapes



class HoverWidget(widgets.VBox):
    
    valueX = traitlets.Any().tag(sync=True)
    valueY = traitlets.Any().tag(sync=True)

    values = traitlets.List([]).tag(sync=True)

    def __init__(self, width=300):
        super(HoverWidget, self).__init__()

        self.width = width

        # output_dummy_img = io.BytesIO()
        # Image.new('RGB', (self.width,self.width)).save(output_dummy_img, format="JPEG")
        # self.img_widgets = {'valueX': widgets.Image(value=output_dummy_img.getvalue(), width=0, height=0), 
        #                     'valueY': widgets.Image(value=output_dummy_img.getvalue(), width=0)} #, height=0)}
        # self.txt_widgets = {'valueX': widgets.HTML(value='', layout=widgets.Layout(width="%ipx"%self.width)), 
        #                     'valueY': widgets.HTML(value='', layout=widgets.Layout(width="%ipx"%self.width)), }
        # self.wav_widgets = {'valueX': widgets.Audio(autoplay=True, layout=widgets.Layout(width="0px", height="0px")),
        #                     'valueY': widgets.Audio(autoplay=True, layout=widgets.Layout(width="0px", height="0px"))}

        # self.children = [widgets.VBox(list(self.wav_widgets.values())), widgets.VBox(list(self.txt_widgets.values())), widgets.VBox(list(self.img_widgets.values()))]

        self.layout = widgets.Layout(width="%ipx"%(self.width+10), height="inherit")


    @traitlets.validate("valueX", "valueY")
    def _validate_value(self, proposal):
        # valueX and valueY are deprecated. use values instead
        self.values = [proposal.value]
        return proposal.value
    
    
    # def set_text(self, name, value):
    #     cur_img_widget = self.img_widgets[name]
    #     cur_txt_widget = self.txt_widgets[name]
    #     cur_wav_widget = self.wav_widgets[name]

    #     cur_txt_widget.value = "<div style='word-wrap: break-word;'>{}</div>".format(value)
    #     cur_img_widget.width = 0
    #     # cur_img_widget.height = 0
    #     cur_wav_widget.value = b'RIFF$\xe2\x04\x00WAVEfmt'
    #     cur_wav_widget.layout = widgets.Layout(width="0px", height="0px")

    # def set_img(self, name, value):
    #     cur_img_widget = self.img_widgets[name]
    #     cur_txt_widget = self.txt_widgets[name]
    #     cur_wav_widget = self.wav_widgets[name]

    #     cur_img_widget.value = value
    #     cur_img_widget.width = self.width
    #     # cur_img_widget.height = self.width
    #     cur_txt_widget.value = ""
    #     cur_wav_widget.value = b'RIFF$\xe2\x04\x00WAVEfmt'
    #     cur_wav_widget.layout = widgets.Layout(width="0px", height="0px")

    # def set_wav(self, name, value):
    #     cur_img_widget = self.img_widgets[name]
    #     cur_txt_widget = self.txt_widgets[name]
    #     cur_wav_widget = self.wav_widgets[name]

    #     cur_img_widget.width = 0
    #     cur_txt_widget.value = ""
    #     cur_wav_widget.value = value
    #     cur_wav_widget.layout = widgets.Layout(width="%ipx"%self.width, height="20px")
    # TODO: make more efficient
    def add_txt(self, value):
        self.children = self.children + (widgets.HTML(value="<div style='word-wrap: break-word;'>{}</div>".format(value), layout=widgets.Layout(width="%ipx"%self.width)),)

    def add_img(self, value):
        self.children = self.children + (widgets.Image(value=value, width=self.width),)
    
    def add_wav(self, value):
        self.children = self.children + (widgets.Audio(value=value, autoplay=True, layout=widgets.Layout(width="%ipx"%self.width, height="20px")),)

    # @traitlets.observe("valueX", "valueY")
    @traitlets.observe("values")
    def onUpdateValue(self, change):
        self.children = []
        
        for value in change.new:
            if type(value) is io.BytesIO:
                self.add_img(value.getvalue())

            elif type(value) is dict:
                if "displayType" in value:
                    if value["displayType"] == am_data.DisplayTypes.IMAGE:
                        self.add_img(value["value"])
                    elif value["displayType"] == am_data.DisplayTypes.AUDIO:
                        self.add_wav(value["value"])
                    else:
                        self.add_txt(value["value"])

                elif "value" in value:
                    self.add_txt(value["value"])
            else:
                self.add_txt(value)
        



from sklearn.decomposition import PCA
from umap import UMAP
available_projection_methods = {
    'PCA': {'module': PCA, 'OOS':False}, # OOS: flag to signal whether or not out of sample is possible
    'UMAP': {'module': UMAP, 'OOS':True},
}
try: 
    from openTSNE import TSNE
    available_projection_methods["TSNE"] = {'module': TSNE, 'OOS':False}
except ImportError:
    print("To support TSNE dataset, please install 'openTSNE': 'pip install openTSNE'.")


class ScatterPlotWidget(widgets.VBox):
    
    embedding = traitlets.Any().tag(sync=True)
    cluster = traitlets.Any().tag(sync=True)
    hover_idcs = traitlets.Any().tag(sync=True)
    mark_colors = ["#ff7f00", "#377eb8", "#4daf4a", "#e41a1c", "#984ea3", "#ffff33", "#a65628", "#f781bf", "#999999"]

    def __init__(self, seed=31415, modality1_label='Image', modality2_label='Text', hover_callback=None, unhover_callback=None):
        super(ScatterPlotWidget, self).__init__()

        self.hover_callback = hover_callback
        self.unhover_callback = unhover_callback
        self.seed=seed

        self.nr_components_widget = widgets.BoundedIntText(
            value=2,
            min=2,
            max=10,
            step=1,
            description='Nr Components:',
            disabled=False,
            layout=widgets.Layout(width="150px")
        )
        self.nr_components_widget.observe(self.onUpdateValue, 'value')

        self.x_component_widget = widgets.BoundedIntText(
            value=1,
            min=1,
            max=10,
            step=1,
            description='X component:',
            disabled=False,
            layout=widgets.Layout(width="150px")
        )
        self.x_component_widget.observe(self.update_scatter, 'value')

        self.y_component_widget = widgets.BoundedIntText(
            value=2,
            min=1,
            max=10,
            step=1,
            description='Y component:',
            disabled=False,
            layout=widgets.Layout(width="150px")
        )
        self.y_component_widget.observe(self.update_scatter, 'value')

        traitlets.dlink((self.nr_components_widget, 'value'), (self.x_component_widget, 'max'))
        traitlets.dlink((self.nr_components_widget, 'value'), (self.y_component_widget, 'max'))


        self.fig_widget = go.FigureWidget()
        self.fig_widget.update_layout(width=400, 
                                      height=300, 
                                      margin=dict(l=10, r=10, t=10, b=10),
                                      legend=dict(
                                            yanchor="top",
                                            y=0.99,
                                            xanchor="left",
                                            x=0.01  
                                            )
                                      )

        self.select_projection_method = widgets.Dropdown(
            description='Method: ',
            value='UMAP',
            options=list(available_projection_methods),
        )
        self.select_projection_method.observe(self._update_projection_method, 'value')

        self.use_oos_projection = widgets.Checkbox(
            value=False,
            description='Use out of sample projection',
            disabled=not available_projection_methods[self.select_projection_method.value]['OOS'],
            indent=False
        )
        self.use_oos_projection.observe(self._update_projection_method, 'value')

        
        self.children = [self.select_projection_method, self.use_oos_projection, self.nr_components_widget, widgets.HBox([self.x_component_widget, self.y_component_widget]), self.fig_widget]
        

    def _update_projection_method(self, change):
        print('', available_projection_methods[self.select_projection_method.value]['OOS'])
        self.use_oos_projection.disabled = not available_projection_methods[self.select_projection_method.value]['OOS']
        self.onUpdateValue(change)

    @traitlets.validate("embedding")
    def _validate_value(self, proposal):
        # print("TODO: validate embedding")
        # for backwards compatibility map array to dict
        if isinstance(proposal.value, np.ndarray):
            print("Deprecation Warning: Setting embedding as concatenated array of modality embeddings is deprecated. Use a dictionary instead.")
            embeddings = {}
            embeddings["Modality1"] = proposal.value[:int(len(proposal.value)/2),:]
            embeddings["Modality2"] = proposal.value[int(len(proposal.value)/2):,:]
            return embeddings
        return proposal.value

    @traitlets.observe("embedding")
    def onUpdateValue(self, change):
        projection_method = available_projection_methods[self.select_projection_method.value]
        # TODO: add UI for distance metric
        if self.select_projection_method.value == 'PCA':
            projection = projection_method['module'](n_components=self.nr_components_widget.value, random_state=self.seed)
        else:
            projection = projection_method['module'](n_components=self.nr_components_widget.value, metric="cosine", random_state=self.seed)
            
        if not self.use_oos_projection.disabled and self.use_oos_projection.value:
            # only possible with umap; fit by one modality only and project other modalities out of sample
            projection.fit(self.embedding[list(self.embedding.keys())[0]])
        else:
            projection = projection.fit(np.concatenate(list(self.embedding.values())))

        embedding_projection = {}
        for modality in self.embedding.keys():
            embedding_projection[modality] = projection.transform(self.embedding[modality])
        self.embedding_projection = embedding_projection

        self.update_scatter(change)

    @traitlets.observe("hover_idcs")
    def hover_changed(self, change):
        shapes = [line for line in self.fig_widget.layout.shapes if line.name == 'pair_connections']

        stacked_embs = np.stack(list(self.embedding_projection.values()), axis=0)
        # r = (stacked_embs.max((0,1)) - stacked_embs.min((0,1))) * 0.01
        range_x = self.fig_widget.layout.xaxis.range[1] - self.fig_widget.layout.xaxis.range[0]
        range_y = self.fig_widget.layout.yaxis.range[1] - self.fig_widget.layout.yaxis.range[0]
        r = (range_x*0.01, range_y*0.01)
        modalities = list(self.embedding_projection.keys())

        for item in self.hover_idcs:
            if type(item) is int:
                # if a line index is given, we highlight the circles and draw lines to the centroid of all modalities of this line
                line_idx = item
                centroid = stacked_embs[:,line_idx,:].mean(axis=0)

                for i in range(len(modalities)):
                    modality = modalities[i]
                    shapes.append(go.layout.Shape(name='hover_line', 
                                                    type='line', 
                                                    x0=self.embedding_projection[modality][line_idx,self.x_component_widget.value-1], 
                                                    y0=self.embedding_projection[modality][line_idx,self.y_component_widget.value-1], 
                                                    x1=centroid[0], 
                                                    y1=centroid[1], 
                                                    line=dict(color="yellow", width=1)))
                    
                    shapes.append(go.layout.Shape(name="hover_circle",
                                                type="circle",
                                                xref="x",
                                                yref="y",
                                                x0=self.embedding_projection[modality][line_idx,self.x_component_widget.value-1]-r[0],
                                                x1=self.embedding_projection[modality][line_idx,self.x_component_widget.value-1]+r[0],
                                                y0=self.embedding_projection[modality][line_idx,self.y_component_widget.value-1]-r[1],
                                                y1=self.embedding_projection[modality][line_idx,self.y_component_widget.value-1]+r[1],
                                                fillcolor="yellow",
                                                line_color="yellow"))
            elif type(item) is tuple:
                # if a tuple of modality and line index is given, we highlight the circles of the given modality and line index
                modality = item[0]
                line_idx = item[1]
                shapes.append(go.layout.Shape(name="hover_circle",
                                            type="circle",
                                            xref="x",
                                            yref="y",
                                            x0=self.embedding_projection[modality][line_idx,self.x_component_widget.value-1]-r[0],
                                            x1=self.embedding_projection[modality][line_idx,self.x_component_widget.value-1]+r[0],
                                            y0=self.embedding_projection[modality][line_idx,self.y_component_widget.value-1]-r[1],
                                            y1=self.embedding_projection[modality][line_idx,self.y_component_widget.value-1]+r[1],
                                            fillcolor="yellow",
                                            line_color="yellow"))


        self.fig_widget.layout.shapes = shapes

    def _on_hover(self, trace, points, state):
        if len(points.point_inds) < 1:
            return
        line_idx = points.point_inds[0]

        self.hover_idcs = [line_idx]

        if self.hover_callback is not None:
            self.hover_callback(trace, points, state)

    def _on_unhover(self, trace, points, state):
        # self.fig_widget.layout.shapes = [line for line in self.fig_widget.layout.shapes if line.name == 'pair_connections']
        if self.unhover_callback is not None:
            self.unhover_callback(trace, points, state)

    def update_scatter(self, change):
        modalities = list(self.embedding_projection.keys())
        if len(self.fig_widget.data) > len(modalities): # remove traces if there are too many
            self.fig_widget.data = self.fig_widget.data[:len(modalities)]
        for i, modality in enumerate(modalities):
            x_data = self.embedding_projection[modality][:,self.x_component_widget.value-1]
            y_data = self.embedding_projection[modality][:,self.y_component_widget.value-1]
            if len(self.fig_widget.data) <= i: # add a new trace
                trace = go.Scatter(name=modality, x=x_data, y=y_data, mode="markers", marker_color=self.mark_colors[i], hoverinfo="text")
                self.fig_widget.add_trace(trace)
                self.fig_widget.data[i].on_hover(self._on_hover)
                self.fig_widget.data[i].on_unhover(self._on_unhover)
            else:
                self.fig_widget.data[i].x = x_data
                self.fig_widget.data[i].y = y_data
                self.fig_widget.data[i].name = modality

        # self.scatter_image.x = self.image_embedding_projection[:,self.x_component_widget.value-1]
        # self.scatter_image.y = self.image_embedding_projection[:,self.y_component_widget.value-1]

        # self.scatter_text.x = self.text_embedding_projection[:,self.x_component_widget.value-1]
        # self.scatter_text.y = self.text_embedding_projection[:,self.y_component_widget.value-1]

        lines = []
        for line_idx in range(len(list(self.embedding_projection.values())[0])):
            
            # for i in range(len(modalities)-1):
            #     modality1 = modalities[i]
            #     modality2 = modalities[i+1]
            #     lines.append(go.layout.Shape(name='pair_connections', 
            #                                  type='line', 
            #                                  x0=self.embedding_projection[modality1][line_idx,self.x_component_widget.value-1], 
            #                                  y0=self.embedding_projection[modality1][line_idx,self.y_component_widget.value-1], 
            #                                  x1=self.embedding_projection[modality2][line_idx,self.x_component_widget.value-1], 
            #                                  y1=self.embedding_projection[modality2][line_idx,self.y_component_widget.value-1], 
            #                                  line=dict(color="grey", width=1)))

            centroid = np.stack(list(self.embedding_projection.values()))[:,line_idx,:].mean(axis=0)
            for i in range(len(modalities)):
                modality = modalities[i]
                lines.append(go.layout.Shape(name='pair_connections', 
                                             type='line', 
                                             x0=self.embedding_projection[modality][line_idx,self.x_component_widget.value-1], 
                                             y0=self.embedding_projection[modality][line_idx,self.y_component_widget.value-1], 
                                             x1=centroid[0], 
                                             y1=centroid[1], 
                                             line=dict(color="grey", width=1),
                                             opacity=0.4,))
            
        self.fig_widget.layout.shapes = lines


    @traitlets.validate("cluster")
    def _validate_cluster(self, proposal):
        # takes a list of cluster labels + sizes
        # print("TODO: validate cluster")
        return proposal.value

    @traitlets.observe("cluster")
    def onUpdateCluster(self, change):
        print(change)




class SimilarityHeatmapClusteringWidget(widgets.VBox):
    
    embedding = traitlets.Dict().tag(sync=True)
    value = traitlets.Any(np.zeros((6,6))).tag(sync=True)
    cluster = traitlets.Any().tag(sync=True)
    modality_labels = traitlets.List([]).tag(sync=True)
    cluster_label_data = None

    hover_idx = traitlets.List([]).tag(sync=True)


    def __init__(self, zmin=None, zmax=None, cluster_label_data=None, hover_callback=None):
        super(SimilarityHeatmapClusteringWidget, self).__init__()

        self.cluster_label_data = cluster_label_data
        self.size = 6
        self.hover_callback = hover_callback

        self.cluster_similarity_matrix_widget = widgets.Checkbox(
            value=False,
            description='Cluster matrix by similarity',
            disabled=False,
            indent=False
        )
        self.cluster_similarity_matrix_by_widget1 = widgets.Dropdown(
            options=self.modality_labels,
            description='between',
            style = {"description_width": "initial"},
            layout=widgets.Layout(width="150px")
        )
        self.cluster_similarity_matrix_by_widget2 = widgets.Dropdown(
            options=self.modality_labels,
            description='and',
            style = {"description_width": "initial"},
            layout=widgets.Layout(width="150px")
        )
        
        self.fig_widget = go.FigureWidget(data=[go.Heatmap(z=self.value, zmin=zmin, zmax=zmax)])
        self.heatmap = self.fig_widget.data[0]
        self.heatmap.hoverinfo = "text"
        self.fig_widget.update_layout(width=500, height=420,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        self.fig_widget.update_yaxes(autorange='reversed', fixedrange=False)
        self.fig_widget.update_xaxes(fixedrange=False)
        self.fig_widget.layout.shapes = self._get_matrix_gridlines()

        self.heatmap.on_hover(self._hover_fn)
        
        self.cluster_similarity_matrix_widget.observe(self.onUpdateEmbedding, names='value')
        self.cluster_similarity_matrix_by_widget1.observe(self.onUpdateClusterBy, names='value')
        self.cluster_similarity_matrix_by_widget2.observe(self.onUpdateClusterBy, names='value')

        settings = widgets.HBox([self.cluster_similarity_matrix_widget, 
                                 self.cluster_similarity_matrix_by_widget1, 
                                 self.cluster_similarity_matrix_by_widget2], 
                                 layout=widgets.Layout(width="450px"))
        self.children = [settings, widgets.HBox([self.fig_widget])]



    def _hover_fn(self, trace, points, state):
        x_idx = points.xs[0]
        y_idx = points.ys[0]
            
        # show hover lines in similarity heatmap
        self.hover_idx = [(x_idx, y_idx)]

        # extract original x and y index and modalities for x and y; then call the callback function
        if self.hover_callback is not None:
            x_modality_idx = math.floor(x_idx/self.size)
            y_modality_idx = math.floor(y_idx/self.size)

            # x_modality = 'modality2'
            # if x_idx < self.size:
            #     x_modality = 'modality1'

            # y_modality = 'modality2'
            # if y_idx < len(self.idcs):
            #     y_modality = 'modality1'

            x_idx = self.idcs[x_idx%len(self.idcs)]
            y_idx = self.idcs[y_idx%len(self.idcs)]
            # self.hover_callback(x_idx, y_idx, x_modality, y_modality)
            self.hover_callback(x_idx, y_idx, self.modality_labels[x_modality_idx], self.modality_labels[y_modality_idx])

    def _get_matrix_gridlines(self):
        no_modalities = len(self.modality_labels)  

        line_style = dict(color="black", width=1)
        gridlines = []
        for i in range(no_modalities-1):
            horizontal_line = go.layout.Shape(type='line', x0=len(self.value)*(1+i)/no_modalities-0.5, y0=0-0.5, x1=len(self.value)*(1+i)/no_modalities-0.5, y1=len(self.value)-0.5, line=line_style)
            gridlines.append(horizontal_line)
            vertical_line = go.layout.Shape(type='line', y0=len(self.value)*(1+i)/no_modalities-0.5, x0=0-0.5, y1=len(self.value)*(1+i)/no_modalities-0.5, x1=len(self.value)-0.5, line=line_style)
            gridlines.append(vertical_line)

        return gridlines

    @traitlets.observe("modality_labels")
    def onUpdateModalityLabel(self, change):
        no_modalities = len(self.modality_labels)
        self.fig_widget.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = [(i*2+1)/2 * self.size for i in range(no_modalities)], #[1*len(self.value)/4, 3*len(self.value)/4]
                ticktext = self.modality_labels
            ),
            yaxis = dict(
                tickmode = 'array',
                tickvals = [(i*2+1)/2 * self.size for i in range(no_modalities)], #[1*len(self.value)/4, 3*len(self.value)/4]
                ticktext = self.modality_labels
            ),
        )

        self.cluster_similarity_matrix_by_widget1.options = self.modality_labels
        if self.cluster_similarity_matrix_by_widget1.value is None:
            self.cluster_similarity_matrix_by_widget1.value = self.modality_labels[0]
        self.cluster_similarity_matrix_by_widget2.options = self.modality_labels
        if self.cluster_similarity_matrix_by_widget2.value is None:
            self.cluster_similarity_matrix_by_widget2.value = self.modality_labels[0]
    

    def onUpdateClusterBy(self, change):
        if self.cluster_similarity_matrix_by_widget1.value is not None and self.cluster_similarity_matrix_by_widget2.value is not None and self.cluster_similarity_matrix_widget.value:
            self.onUpdateEmbedding(change)


    @traitlets.validate("value")
    def _validate_value(self, proposal):
        # print("TODO: validate value")
        return proposal.value

    @traitlets.observe("value")
    def onUpdateValue(self, change):
        self.fig_widget.data[0].z = self.value
        self.fig_widget.layout.shapes = self._get_matrix_gridlines()

    @traitlets.validate("embedding")
    def _validate_embedding(self, proposal):
        # print("TODO: validate embedding")
        # for backwards compatibility map array to dict
        if isinstance(proposal.value, tuple):
            print("Deprecation Warning: Setting embedding as tuple of modality embeddings is deprecated. Use a dictionary instead.")
            embeddings = {}
            embeddings["Modality1"] = proposal.value[0]
            embeddings["Modality2"] = proposal.value[1]
            return embeddings
        return proposal.value

    @traitlets.observe("embedding")
    def onUpdateEmbedding(self, change):
        self.size=len(list(self.embedding.values())[0])
        self.modality_labels = list(self.embedding.keys())
        no_modalities = len(self.embedding.keys())
        similarity_all = get_similarities_all(self.embedding)

        cluster_labels = []
        cluster_sizes = []

        if self.cluster_similarity_matrix_widget.value:
            similarity_between = self.embedding[self.cluster_similarity_matrix_by_widget1.value]
            similarity_and = self.embedding[self.cluster_similarity_matrix_by_widget2.value]
            similarity_cross_modal = get_similarities(torch.from_numpy(similarity_between), torch.from_numpy(similarity_and))
            self.idcs, clusters, clusters_unsorted = get_cluster_sorting(similarity_cross_modal)
            for c in set(clusters):
                cluster_size = np.count_nonzero(clusters==c)
                cluster_label = ""
                if self.cluster_label_data is not None:
                    cluster_label = self.cluster_label_data.getMinSummary(np.where(clusters_unsorted==c)[0])
                cluster_labels.append(cluster_label)
                cluster_sizes.append(cluster_size)
        else:
            self.idcs = np.arange(self.size) # TODO: use reverse idcs to get original order for interaction with other widgets

        # with heatmap_widget.batch_update():
        matrix_sort_idcs = np.concatenate([self.idcs + i*self.size for i in range(no_modalities)], axis=0) # [self.idcs, self.idcs+self.size] # need to do double index because we combined images and texts
        
        self.value = similarity_all[matrix_sort_idcs, :][:, matrix_sort_idcs]
        self.cluster = (cluster_labels, cluster_sizes)

        
    # @traitlets.validate("cluster")
    # def _validate_cluster(self, proposal):
        # takes a list of cluster labels + sizes
    #     print("TODO: validate cluster")
    #     return proposal.value

    @traitlets.observe("cluster")
    def onUpdateCluster(self, change):
        cluster_modality1_idx = self.modality_labels.index(self.cluster_similarity_matrix_by_widget1.value)
        cluster_modality2_idx = self.modality_labels.index(self.cluster_similarity_matrix_by_widget2.value)
        cluster_shapes = self._get_matrix_gridlines()
        labels, sizes = self.cluster
        offset = 0-0.5 # -0.5 because heatmap rectangles are drawn around [-0.5, 0.5]
        for (cluster_label, cluster_size) in zip(labels, sizes):
            if cluster_size > 5:
                textposition = 'middle left' if offset < self.size/2 else 'middle right'

                # see https://plotly.com/python/shapes/
                cluster_shapes += [go.layout.Shape(
                    type='rect', 
                    x0=cluster_modality1_idx*self.size+offset, 
                    y0=cluster_modality2_idx*self.size+offset, 
                    x1=cluster_modality1_idx*self.size+offset+cluster_size, 
                    y1=cluster_modality2_idx*self.size+offset+cluster_size, 
                    label=dict(text=cluster_label, textposition=textposition, font=dict(size=10, color="white"), padding=np.log(cluster_size)*10), 
                    line=dict(width=1, color='white')
                )]

            offset += cluster_size
            
        self.fig_widget.layout.shapes = cluster_shapes


    @traitlets.observe("hover_idx")
    def onUpdateHoverIdx(self, change):
        shapes = [sh for sh in self.fig_widget.layout.shapes if sh.name != 'hover_idx' and sh.name != 'hover_idx']

        for (x_idx, y_idx) in self.hover_idx:
            if x_idx >= 0 and x_idx < len(self.value):
                shapes.append(go.layout.Shape(name='hover_idx', type='line', x0=x_idx, y0=0-0.5, x1=x_idx, y1=len(self.value)-0.5, line=dict(color="grey", width=1)))
            if y_idx >= 0 and y_idx < len(self.value):
                shapes.append(go.layout.Shape(name='hover_idx', type='line', y0=y_idx, x0=0-0.5, y1=y_idx, x1=len(self.value)-0.5, line=dict(color="grey", width=1)))
        
        self.fig_widget.layout.shapes = shapes




class CLIPExplorerWidget(widgets.AppLayout):
    def __init__(self, dataset_name, all_images=None, all_prompts=None, all_data=None, models=None):
        # using "all_images" and "all_prompts" is deprecated; use the "all_data" dict instead; e.g. {"image": all_images, "text": all_prompts}
        ### models... list of strings or instances that inherit from CLIPModelInterface 
        super(CLIPExplorerWidget, self).__init__()
        
        # for backwards compatibility
        if all_data is None:
            all_data = {}
            print('WARNING: using all_images and all_prompts is deprecated; use the "all_data" dict instead; e.g. {"image": all_images, "text": all_prompts}')
            if all_images is not None:
                all_data["image"] = all_images
            if all_prompts is not None:
                all_data["text"] = all_prompts

        if models is None:
            models = am_model.available_CLIP_models

        self.models = {}
        for m in models:
            if type(m) == str:
                self.models[m] = am_model.get_model(m)
            elif issubclass(type(m), am_model.CLIPModelInterface):
                self.models[m.model_name] = m
            else:
                print('skipped', m, 'because it is not string or of type CLIPModelInterface')

        
        self.dataset_name = dataset_name
        self.all_data = all_data
        self.size = len(all_data[list(all_data.keys())[0]])

        # ui select widgets
        self.model_select_widget = widgets.Dropdown(
            description='Model: ',
            value=list(self.models.keys())[0],
            options=list(self.models.keys()),
        )

        m = self.models[self.model_select_widget.value]
        self.available_modalities = set(list(self.all_data.keys())).intersection(m.encoding_functions.keys())
        

        self.close_modality_gap_widget = widgets.Checkbox(
            value=False,
            description='Close modality gap',
            disabled=False,
            indent=False
        )

        self.modality1_select_widget = widgets.Dropdown(
            description='Modality 1: ',
            value=list(self.available_modalities)[0],
            options=list(self.available_modalities),
        )
        
        self.modality2_select_widget = widgets.Dropdown(
            description='Modality 2: ',
            value=list(self.available_modalities)[1],
            options=list(self.available_modalities),
        )
        
        # TODO: make dynamic -> users should be able to add/remove modalities dynamically
        if len(self.available_modalities) != 2:
            self.close_modality_gap_widget.layout.visibility = 'hidden'
        
            modality1_data = list(self.all_data.values())[0]
        else:
            modality1_data = self.all_data[self.modality1_select_widget.value]
            # modality2_data = self.all_data[self.modality2_select_widget.value]

        # output widgets
        self.hover_widget = HoverWidget()

        self.embeddings, self.logit_scale = get_embeddings_per_modality(m, self.dataset_name, self.all_data)
        self.scatter_widget = ScatterPlotWidget(hover_callback=self.scatter_hover_fn, unhover_callback=self.scatter_unhover_fn)#(modality1_label=modality1_data.name, modality2_label=modality2_data.name)
        
        self.heatmap_widget = SimilarityHeatmapClusteringWidget(
            cluster_label_data=modality1_data, 
            hover_callback=self.heatmap_hover_fn)
        
        self.log_widget = widgets.Output()

        # TODO: make dynamic -> users should be able to add/remove modalities dynamically
        if len(self.available_modalities) == 2:
            embedding_modality1 = self.embeddings[self.modality1_select_widget.value]
            embedding_modality2 = self.embeddings[self.modality2_select_widget.value]
            self.scatter_widget.embedding = {self.modality1_select_widget.value: embedding_modality1, self.modality2_select_widget.value: embedding_modality2} 
            # self.scatter_widget.embedding = np.concatenate((embedding_modality1, embedding_modality2))

            # TODO: calculate this for all modality combinations
            modality_distance = get_modality_distance(embedding_modality1, embedding_modality2)
            validation_loss = calculate_val_loss(embedding_modality1, embedding_modality2, self.logit_scale.exp())
            with self.log_widget:
                print('Modality distance: %.2f | Loss: %.2f'%(modality_distance, validation_loss))

            self.heatmap_widget.embedding = {self.modality1_select_widget.value: embedding_modality1, self.modality2_select_widget.value: embedding_modality2} #np.concatenate((embedding_modality1, embedding_modality2))
        else:
            embedding_modality1 = self.embeddings[self.modality1_select_widget.value]
            embedding_modality2 = self.embeddings[self.modality2_select_widget.value]
            self.scatter_widget.embedding = self.embeddings
            self.heatmap_widget.embedding = self.embeddings
            with self.log_widget:
                print("ready")

        # callback functions
        self.model_select_widget.observe(self.model_changed, names="value")
        self.close_modality_gap_widget.observe(self.model_changed, names='value')
        self.modality1_select_widget.observe(self.modality_changed, names="value")
        self.modality2_select_widget.observe(self.modality_changed, names="value")

        # display everyting
        header_list = []
        header_list.append(widgets.HBox([self.model_select_widget, self.close_modality_gap_widget]))
        if len(self.available_modalities) == 2: # TODO: make dynamic -> users should be able to add/remove modalities dynamically
            header_list.append(widgets.HBox([self.modality1_select_widget, self.modality2_select_widget]))
        header_list.append(self.log_widget)
        
        self.header = widgets.VBox(header_list)
        self.header.layout.height = '%ipx'%(40*len(header_list))
        vis_widgets = widgets.HBox([self.heatmap_widget, self.scatter_widget])
        self.center = vis_widgets
        self.right_sidebar = self.hover_widget
        self.height = '700px'


    def model_changed(self, change):

        self.log_widget.clear_output()
        with self.log_widget:
            print("loading...")

        m = self.models[self.model_select_widget.value]
        self.available_modalities = set(list(self.all_data.keys())).intersection(m.encoding_functions.keys())
        self.modality1_select_widget.options = list(self.available_modalities)
        self.modality2_select_widget.options = list(self.available_modalities)

        self.embeddings, self.logit_scale = get_embeddings_per_modality(m, self.dataset_name, self.all_data)
        
        # TODO: make dynamic -> users should be able to add/remove modalities dynamically
        if len(self.available_modalities) == 2:
            self.modality_changed(change)
        else:
            self.scatter_widget.embedding = self.embeddings
            self.heatmap_widget.embedding = self.embeddings
            self.log_widget.clear_output()
            with self.log_widget:
                print('ready')



    def modality_changed(self, change):
        self.log_widget.clear_output()
        with self.log_widget:
            print("loading...")
            
        modality1_data = self.all_data[self.modality1_select_widget.value]
        modality2_data = self.all_data[self.modality2_select_widget.value]

        # self.scatter_widget.modality1_label = modality1_data.name
        # self.scatter_widget.modality2_label = modality2_data.name
        # self.heatmap_widget.modality1_label = modality1_data.name
        # self.heatmap_widget.modality2_label = modality2_data.name
        self.heatmap_widget.cluster_label_data = modality1_data

        embedding_modality1 = self.embeddings[self.modality1_select_widget.value]
        embedding_modality2 = self.embeddings[self.modality2_select_widget.value]

        if self.close_modality_gap_widget.value:
            embedding_modality1, embedding_modality2 = get_closed_modality_gap(embedding_modality1, embedding_modality2)
            # image_embedding, text_embedding = get_closed_modality_gap_rotated(image_embedding, text_embedding)

        # self.scatter_widget.embedding = np.concatenate((embedding_modality1, embedding_modality2))
        self.scatter_widget.embedding = {self.modality1_select_widget.value: embedding_modality1, self.modality2_select_widget.value: embedding_modality2}
        # self.heatmap_widget.embedding = np.concatenate((embedding_modality1, embedding_modality2))
        self.heatmap_widget.embedding = {self.modality1_select_widget.value: embedding_modality1, self.modality2_select_widget.value: embedding_modality2}

        modality_distance = get_modality_distance(embedding_modality1, embedding_modality2)
        # modality_distance = get_modality_distance_rotated(image_embedding, text_embedding)
        
        validation_loss = calculate_val_loss(embedding_modality1, embedding_modality2, self.logit_scale.exp())

        self.log_widget.clear_output()
        with self.log_widget:
            print('Modality distance: %.2f | Loss: %.2f'%(modality_distance, validation_loss))


    def heatmap_hover_fn(self, x_idx, y_idx, x_modality, y_modality):
        self.hover_widget.valueX = self.all_data[x_modality].getVisItem(x_idx)
        self.hover_widget.valueY = self.all_data[y_modality].getVisItem(y_idx)
        self.hover_widget.values = [self.all_data[x_modality].getVisItem(x_idx), self.all_data[y_modality].getVisItem(y_idx)]
        # modality1_data = self.all_data[self.modality1_select_widget.value]
        # modality2_data = self.all_data[self.modality2_select_widget.value]
        # # show hover images/texts
        # if x_modality == "modality1":
        #     self.hover_widget.valueX = modality1_data.getVisItem(x_idx)
        # else:
        #     self.hover_widget.valueX = modality2_data.getVisItem(x_idx)
        
        # if y_modality == "modality1":
        #     self.hover_widget.valueY = modality1_data.getVisItem(y_idx)
        # else:
        #     self.hover_widget.valueY = modality2_data.getVisItem(y_idx)

        self.scatter_widget.hover_idcs = [(x_modality, x_idx), (y_modality, y_idx)]

    def scatter_hover_fn(self, trace, points, state):
        if len(points.point_inds) < 1:
            return
        idx = points.point_inds[0]
        # print(trace.name, idx) # image vs text trace
        # modality1_data = self.all_data[self.modality1_select_widget.value]
        # modality2_data = self.all_data[self.modality2_select_widget.value]

        # self.hover_widget.valueY = modality1_data.getVisItem(idx)
        # self.hover_widget.valueX = modality2_data.getVisItem(idx)
        self.hover_widget.values = [self.all_data[modality].getVisItem(idx) for modality in self.all_data.keys()]
        
        inverse_idcs = np.argsort(self.heatmap_widget.idcs)
        heatmap_idx = inverse_idcs[idx]
        # self.heatmap_widget.hover_idx = [(heatmap_idx, self.size + heatmap_idx), (self.size + heatmap_idx, heatmap_idx)]
        self.heatmap_widget.hover_idx = [(heatmap_idx + i*self.size, heatmap_idx + i*self.size) for i in range(len(self.available_modalities))]


    def scatter_unhover_fn(self, trace, points, state):
        self.heatmap_widget.hover_idx = []


class CLIPExplorerWidget_Old(widgets.AppLayout):
    idcs = traitlets.Any().tag(sync=True)

    def __init__(self, dataset_name, all_images, all_prompts, models=None):
        ### models... list of strings or instances that inherit from CLIPModelInterface 
        super(CLIPExplorerWidget_Old, self).__init__()

        if models is None:
            models = am_model.available_CLIP_models

        self.models = {}
        for m in models:
            if type(m) == str:
                self.models[m] = am_model.get_model(m)
            elif issubclass(type(m), am_model.CLIPModelInterface):
                self.models[m.model_name] = m
            else:
                print('skipped', m, 'because it is not string or of type CLIPModelInterface')

        
        self.dataset_name = dataset_name
        self.all_images = np.array(all_images)
        self.all_prompts = np.array(all_prompts)
        self.size = len(all_images)
        self.idcs = np.arange(self.size)

        # ui select widgets
        self.model_select_widget = widgets.Dropdown(
            description='Model: ',
            value=list(self.models.keys())[0],
            options=list(self.models.keys()),
        )

        self.cluster_similarity_matrix_widget = widgets.Checkbox(
            value=False,
            description='Cluster matrix by similarity',
            disabled=False,
            indent=False
        )

        self.close_modality_gap_widget = widgets.Checkbox(
            value=False,
            description='Close modality gap',
            disabled=False,
            indent=False
        )

        # output widgets
        self.hover_widget = HoverWidget()

        m = self.models[self.model_select_widget.value]
        image_embedding_norm, text_embedding_norm, logit_scale = get_embedding(m, self.dataset_name, self.all_images, self.all_prompts)
        self.scatter_widget = ScatterPlotWidget()
        self.scatter_widget.embedding = np.concatenate((image_embedding_norm, text_embedding_norm))

        modality_distance = get_modality_distance(image_embedding_norm, text_embedding_norm)
        validation_loss = calculate_val_loss(image_embedding_norm, text_embedding_norm, logit_scale.exp())
        self.log_widget = widgets.Output()
        with self.log_widget:
            print('Modality distance: %.2f | Loss: %.2f'%(modality_distance, validation_loss))

        similarity_texts_images, similarity_all = get_similarity(image_embedding_norm, text_embedding_norm)
        self.heatmap_widget = SimilarityHeatmapWidget()
        self.heatmap_widget.value = similarity_all



        # callback functions
        self.model_select_widget.observe(self.model_changed, names="value")
        self.cluster_similarity_matrix_widget.observe(self.model_changed, names='value')
        self.close_modality_gap_widget.observe(self.model_changed, names='value')
        self.heatmap_widget.heatmap.on_hover(self.hover_fn)
        self.scatter_widget.scatter_image.on_hover(self.scatter_hover_fn)
        self.scatter_widget.scatter_text.on_hover(self.scatter_hover_fn)
        self.scatter_widget.scatter_image.on_unhover(self.scatter_unhover_fn)
        self.scatter_widget.scatter_text.on_unhover(self.scatter_unhover_fn)

        # display everyting
        self.header = widgets.VBox([widgets.HBox([self.model_select_widget, self.cluster_similarity_matrix_widget, self.close_modality_gap_widget]),self.log_widget])
        self.header.layout.height = '80px'
        vis_widgets = widgets.HBox([self.heatmap_widget, self.scatter_widget])
        self.center = vis_widgets
        self.right_sidebar = self.hover_widget
        self.height = '700px'


    def model_changed(self, change):

        self.log_widget.clear_output()
        with self.log_widget:
            print("loading...")

        m = self.models[self.model_select_widget.value]
        image_embedding, text_embedding, logit_scale = get_embedding(m, self.dataset_name, self.all_images, self.all_prompts)

        if self.close_modality_gap_widget.value:
            image_embedding, text_embedding = get_closed_modality_gap(image_embedding, text_embedding)
            # image_embedding, text_embedding = get_closed_modality_gap_rotated(image_embedding, text_embedding)

        self.scatter_widget.embedding = np.concatenate((image_embedding, text_embedding))

        modality_distance = get_modality_distance(image_embedding, text_embedding)
        # modality_distance = get_modality_distance_rotated(image_embedding, text_embedding)
        
        validation_loss = calculate_val_loss(image_embedding, text_embedding, logit_scale.exp())
        
        similarity_texts_images, similarity_all = get_similarity(image_embedding, text_embedding)

        cluster_labels = []
        cluster_sizes = []

        if self.cluster_similarity_matrix_widget.value:
            self.idcs, clusters, clusters_unsorted = get_cluster_sorting(similarity_texts_images)
            for c in set(clusters):
                cluster_size = np.count_nonzero(clusters==c)
                cluster_label = get_textual_label_for_cluster(np.where(clusters_unsorted==c)[0], self.all_prompts)
                cluster_labels.append(cluster_label)
                cluster_sizes.append(cluster_size)
        else:
            self.idcs = np.arange(self.size)

        # with heatmap_widget.batch_update():
        matrix_sort_idcs = np.concatenate([self.idcs, self.idcs+self.size], axis=0) # need to do double index because we combined images and texts
        self.heatmap_widget.value = similarity_all[matrix_sort_idcs, :][:, matrix_sort_idcs]
        self.heatmap_widget.cluster = (cluster_labels, cluster_sizes)

        self.log_widget.clear_output()
        with self.log_widget:
            print('Modality distance: %.2f | Loss: %.2f'%(modality_distance, validation_loss))


    def hover_fn(self, trace, points, state):
        x_idx = points.xs[0]
        y_idx = points.ys[0]
            
        self.heatmap_widget.hover_idx = [(x_idx, y_idx)]

        if x_idx < self.size:
            output_img = io.BytesIO()
            self.all_images[self.idcs][x_idx].resize((300,300)).save(output_img, format='JPEG')
            self.hover_widget.valueX = output_img
        else:
            self.hover_widget.valueX = self.all_prompts[self.idcs][x_idx%self.size]
        
        if y_idx < self.size:
            output_img = io.BytesIO()
            self.all_images[self.idcs][y_idx].resize((300,300)).save(output_img, format='JPEG')
            self.hover_widget.valueY = output_img
        else:
            self.hover_widget.valueY = self.all_prompts[self.idcs][y_idx%self.size]

    def scatter_hover_fn(self, trace, points, state):
        if len(points.point_inds) < 1:
            return
        idx = points.point_inds[0]
        # print(trace.name, idx) # image vs text trace

        self.hover_widget.valueX = self.all_prompts[idx]

        output_img = io.BytesIO()
        self.all_images[idx].resize((300,300)).save(output_img, format='JPEG')
        self.hover_widget.valueY = output_img
        
        inverse_idcs = np.argsort(self.idcs)
        heatmap_idx = inverse_idcs[idx]
        self.heatmap_widget.hover_idx = [(heatmap_idx, self.size + heatmap_idx), (self.size + heatmap_idx, heatmap_idx)]


    def scatter_unhover_fn(self, trace, points, state):
        self.heatmap_widget.hover_idx = []




class CLIPComparerWidget(widgets.AppLayout):


    def __init__(self, dataset_name, all_images, all_prompts, models=list(am_model.available_CLIP_models), close_modality_gap=False, zmin=None, zmax=None):
        super(CLIPComparerWidget, self).__init__()
        # close_modality_gap: boolean or list of booleans with same length as models that specifies whether or not the modality gap should be closed
        ### models... list of strings or instances that inherit from CLIPModelInterface 

        if models is None:
            models = am_model.available_CLIP_models

        self.models = {}
        for m in models:
            if type(m) == str:
                modifier = ''
                if m in self.models.keys():
                    modifier = '_%i'%m.__hash__()
                self.models[m+modifier] = am_model.get_model(m)
            elif issubclass(type(m), am_model.CLIPModelInterface):
                modifier = ''
                if m.model_name in self.models.keys():
                    modifier = '_%i'%m.__hash__()
                self.models[m.model_name+modifier] = m
            else:
                print('skipped', m, 'because it is not string or of type CLIPModelInterface')

        

        if type(close_modality_gap) == bool:
            close_modality_gap = [close_modality_gap] * len(self.models)
        print(type(close_modality_gap) == list, close_modality_gap, self.models)
        assert type(close_modality_gap) == list and len(close_modality_gap) == len(self.models), 'close_modality_gap must be a bool or list of the same length as models'
        
        self.dataset_name = dataset_name
        self.all_images = np.array(all_images)
        self.all_prompts = np.array(all_prompts)
        self.size = len(all_images)
        
        # output widgets
        self.hover_widget = HoverWidget()

        self.heatmap_grid = widgets.GridspecLayout(math.ceil(len(self.models)/2), 2)
        for i in range(len(self.models.keys())):
            key = list(self.models.keys())[i]
            m = self.models[key]

            image_embedding_norm, text_embedding_norm, logit_scale = get_embedding(m, self.dataset_name, self.all_images, self.all_prompts)
            if close_modality_gap[i]:
                image_embedding_norm, text_embedding_norm = get_closed_modality_gap(image_embedding_norm, text_embedding_norm)

            _, similarity_all = get_similarity(image_embedding_norm, text_embedding_norm)
            heatmap_widget = SimilarityHeatmapWidget(zmin=zmin, zmax=zmax)
            heatmap_widget.value = similarity_all
            heatmap_widget.heatmap.on_hover(self.hover_fn)
            # heatmap_widget.layout = widgets.Layout(height='500px', width='auto')

            modality_distance = get_modality_distance(image_embedding_norm, text_embedding_norm)
            validation_loss = calculate_val_loss(image_embedding_norm, text_embedding_norm, logit_scale.exp())
            text_widget = widgets.HTML(value='<h2>' + m.model_name + '</h2>' 
                                       + ' Modality distance: %.2f | Loss: %.2f'%(modality_distance, validation_loss))

            self.heatmap_grid[int(i/2), i%2] = widgets.VBox([text_widget, heatmap_widget])
            self.heatmap_grid[int(i/2), i%2].layout.height = '500px'

        self.right_sidebar = self.hover_widget
        self.center = self.heatmap_grid
        self.height = '900px'


    def hover_fn(self, trace, points, state):
        x_idx = points.xs[0]
        y_idx = points.ys[0]
            
        for c in self.heatmap_grid.children:
            c.children[1].hover_idx = [(x_idx, y_idx)]

        if x_idx < self.size:
            output_img = io.BytesIO()
            self.all_images[x_idx].resize((300,300)).save(output_img, format='JPEG')
            valueX = output_img
        else:
            valueX = self.all_prompts[x_idx%self.size]
        
        if y_idx < self.size:
            output_img = io.BytesIO()
            self.all_images[y_idx].resize((300,300)).save(output_img, format='JPEG')
            valueY = output_img
        else:
            valueY = self.all_prompts[y_idx%self.size]

        self.hover_widget.values = [valueX, valueY]

class ModalityGapWidget(widgets.AppLayout):
    
    def __init__(self, image_embedding, text_embedding, logit_scale=100.0, title='Loss Landscape CLIP'):
        super(ModalityGapWidget, self).__init__()
        
        image_embedding = np.array(image_embedding)
        text_embedding = np.array(text_embedding)

        modality_gap = get_modality_gap_normed(image_embedding, text_embedding)
        
        distance_lst = []
        loss_lst = []
        for delta in np.arange(-5.0, 5.0, 0.25): 
            modified_text_features = l2_norm(text_embedding) + 0.5 * delta * modality_gap
            modified_text_features = l2_norm(modified_text_features)

            modified_image_features = l2_norm(image_embedding) - 0.5 * delta * modality_gap
            modified_image_features = l2_norm(modified_image_features)

            avg_val_loss = calculate_val_loss(torch.from_numpy(modified_image_features), torch.from_numpy(modified_text_features), logit_scale = logit_scale)

            pca = PCA(n_components=6)
            pca.fit(np.concatenate((image_embedding, text_embedding), axis=0))

            gap_direction = get_gap_direction(modified_image_features, modified_text_features, pca)

            loss_lst.append(avg_val_loss)

            # Euclidean distance between mass centers
            distance_lst.append(
                get_modality_distance(modified_image_features, modified_text_features) * gap_direction
            )


        orig_distance = get_modality_distance(image_embedding, text_embedding)

        fig = go.FigureWidget(data=go.Scatter(x=distance_lst, y=loss_lst, mode='lines+markers', hovertemplate='Distance: %{x} <br>Loss: %{y}'))
        fig.add_shape(type="line",
            x0=orig_distance, y0=0, x1=orig_distance, y1=max(loss_lst)*1.2,
            line=dict(
                color="Black",
                width=1,
                dash="dash",
            )
        )
        fig.update_layout(xaxis_title='Euclidean Distance', yaxis_title='Loss', width=500, title=title)
        
        self.center = widgets.HBox([fig])
        