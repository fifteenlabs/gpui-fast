#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

// Whether the pipeline this function was specialized into draws clipped
// primitives.
//
// Two variants of every clippable pipeline are compiled out of this one source.
// The unclipped one carries no clip varying, no clip binding and no branch,
// because `if (CLIPPED)` and every argument marked with this constant are
// folded away at specialization time. That is what keeps an arbitrary-path
// clip - which nothing in gpui's own UI uses - free for a frame that never
// asks for one. See `ClipId` and `MetalRenderer::clip_resources`.
constant bool CLIPPED [[function_constant(0)]];

float4 hsla_to_rgba(Hsla hsla);
float3 srgb_to_linear(float3 color);
float3 linear_to_srgb(float3 color);
float4 srgb_to_oklab(float4 color);
float4 oklab_to_srgb(float4 color);
float4 to_device_position(float2 unit_vertex, Bounds_ScaledPixels bounds,
                          constant Size_DevicePixels *viewport_size);
float2 to_scene_position(float2 unit_vertex, Bounds_ScaledPixels bounds);
float2 to_scene_position_transformed(float2 unit_vertex,
                                     Bounds_ScaledPixels bounds,
                                     TransformationMatrix transformation);
float4 to_target_position(float2 scene_position,
                          constant RenderTarget *render_target);
float4 apply_color_matrices(float4 premultiplied, constant GroupFilter *filter);

float2 to_tile_position(float2 unit_vertex, AtlasTile tile,
                        constant Size_DevicePixels *atlas_size);
uint extend_texel(int texel, uint extent, BrushExtend extend);
float4 sample_path_brush(PathBrush brush, float2 position,
                         texture2d<float> atlas);
float4 distance_from_clip_rect(float2 unit_vertex, Bounds_ScaledPixels bounds,
                               Bounds_ScaledPixels clip_bounds);
float4 distance_from_clip_rect_transformed(float2 unit_vertex, Bounds_ScaledPixels bounds,
                               Bounds_ScaledPixels clip_bounds, TransformationMatrix transformation);
float corner_dash_velocity(float dv1, float dv2);
float dash_alpha(float t, float period, float length, float dash_velocity,
                 float antialias_threshold);
float quarter_ellipse_sdf(float2 point, float2 radii);
float pick_corner_radius(float2 center_to_point, Corners_ScaledPixels corner_radii);
float quad_sdf(float2 point, Bounds_ScaledPixels bounds,
               Corners_ScaledPixels corner_radii);
float quad_sdf_impl(float2 center_to_point, float corner_radius);
float content_mask_alpha(float2 point, ContentMask_ScaledPixels mask);
float packed_content_mask_alpha(float2 point, float4 mask_bounds,
                                float4 mask_corner_radii);
float clip_mask_alpha(float2 point, uint clip_id,
                      constant ClipMask *clip_masks,
                      texture2d<float> clip_atlas);
float4 pack_bounds(Bounds_ScaledPixels bounds);
float4 pack_corner_radii(Corners_ScaledPixels corner_radii);
float gaussian(float x, float sigma);
float2 erf(float2 x);
float blur_along_x(float x, float y, float sigma, float corner,
                   float2 half_size);
float4 over(float4 below, float4 above);
float radians(float degrees);
float4 fill_color(Background background, float2 position, Bounds_ScaledPixels bounds,
  float4 solid_color, float4 color0, float4 color1);

struct GradientColor {
  float4 solid;
  float4 color0;
  float4 color1;
};
GradientColor prepare_fill_color(uint tag, uint color_space, Hsla solid, Hsla color0, Hsla color1);

struct QuadVertexOutput {
  uint quad_id [[flat]];
  float4 position [[position]];
  float2 scene_position;
  float4 border_color [[flat]];
  float4 background_solid [[flat]];
  float4 background_color0 [[flat]];
  float4 background_color1 [[flat]];
  float clip_distance [[clip_distance]][4];
};

struct QuadFragmentInput {
  uint quad_id [[flat]];
  float4 position [[position]];
  float2 scene_position;
  float4 border_color [[flat]];
  float4 background_solid [[flat]];
  float4 background_color0 [[flat]];
  float4 background_color1 [[flat]];
};

vertex QuadVertexOutput quad_vertex(uint unit_vertex_id [[vertex_id]],
                                    uint quad_id [[instance_id]],
                                    constant float2 *unit_vertices
                                    [[buffer(QuadInputIndex_Vertices)]],
                                    constant Quad *quads
                                    [[buffer(QuadInputIndex_Quads)]],
                                    constant RenderTarget *render_target
                                    [[buffer(QuadInputIndex_RenderTarget)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  Quad quad = quads[quad_id];
  float2 scene_position = to_scene_position(unit_vertex, quad.bounds);
  float4 device_position = to_target_position(scene_position, render_target);
  float4 clip_distance = distance_from_clip_rect(unit_vertex, quad.bounds,
                                                 quad.content_mask.bounds);
  float4 border_color = hsla_to_rgba(quad.border_color);

  GradientColor gradient = prepare_fill_color(
    quad.background.tag,
    quad.background.color_space,
    quad.background.solid,
    quad.background.colors[0].color,
    quad.background.colors[1].color
  );

  return QuadVertexOutput{
      quad_id,
      device_position,
      scene_position,
      border_color,
      gradient.solid,
      gradient.color0,
      gradient.color1,
      {clip_distance.x, clip_distance.y, clip_distance.z, clip_distance.w}};
}

fragment float4 quad_fragment(QuadFragmentInput input [[stage_in]],
                              constant Quad *quads
                              [[buffer(QuadInputIndex_Quads)]],
                              constant ClipMask *clip_masks
                              [[buffer(QuadInputIndex_ClipMasks),
                                function_constant(CLIPPED)]],
                              texture2d<float> clip_atlas
                              [[texture(QuadInputIndex_ClipAtlas),
                                function_constant(CLIPPED)]]) {
  Quad quad = quads[input.quad_id];
  float mask_alpha = content_mask_alpha(input.scene_position, quad.content_mask);
  if (CLIPPED) {
    mask_alpha *=
        clip_mask_alpha(input.scene_position, quad.clip, clip_masks, clip_atlas);
  }
  float4 background_color = fill_color(quad.background, input.scene_position, quad.bounds,
    input.background_solid, input.background_color0, input.background_color1);

  bool unrounded = quad.corner_radii.top_left == 0.0 &&
    quad.corner_radii.bottom_left == 0.0 &&
    quad.corner_radii.top_right == 0.0 &&
    quad.corner_radii.bottom_right == 0.0;

  // Fast path when the quad is not rounded and doesn't have any border
  if (quad.border_widths.top == 0.0 &&
      quad.border_widths.left == 0.0 &&
      quad.border_widths.right == 0.0 &&
      quad.border_widths.bottom == 0.0 &&
      unrounded) {
    background_color.a *= mask_alpha;
    return background_color;
  }

  float2 size = float2(quad.bounds.size.width, quad.bounds.size.height);
  float2 half_size = size / 2.0;
  float2 point = input.scene_position - float2(quad.bounds.origin.x, quad.bounds.origin.y);
  float2 center_to_point = point - half_size;

  // Signed distance field threshold for inclusion of pixels. 0.5 is the
  // minimum distance between the center of the pixel and the edge.
  const float antialias_threshold = 0.5;

  // Radius of the nearest corner
  float corner_radius = pick_corner_radius(center_to_point, quad.corner_radii);

  // Width of the nearest borders
  float2 border = float2(
    center_to_point.x < 0.0 ? quad.border_widths.left : quad.border_widths.right,
    center_to_point.y < 0.0 ? quad.border_widths.top : quad.border_widths.bottom
  );

  // 0-width borders are reduced so that `inner_sdf >= antialias_threshold`.
  // The purpose of this is to not draw antialiasing pixels in this case.
  float2 reduced_border = float2(
    border.x == 0.0 ? -antialias_threshold : border.x,
    border.y == 0.0 ? -antialias_threshold : border.y);

  // Vector from the corner of the quad bounds to the point, after mirroring
  // the point into the bottom right quadrant. Both components are <= 0.
  float2 corner_to_point = fabs(center_to_point) - half_size;

  // Vector from the point to the center of the rounded corner's circle, also
  // mirrored into bottom right quadrant.
  float2 corner_center_to_point = corner_to_point + corner_radius;

  // Whether the nearest point on the border is rounded
  bool is_near_rounded_corner =
    corner_center_to_point.x >= 0.0 &&
    corner_center_to_point.y >= 0.0;

  // Vector from straight border inner corner to point.
  //
  // 0-width borders are turned into width -1 so that inner_sdf is > 1.0 near
  // the border. Without this, antialiasing pixels would be drawn.
  float2 straight_border_inner_corner_to_point = corner_to_point + reduced_border;

  // Whether the point is beyond the inner edge of the straight border
  bool is_beyond_inner_straight_border =
    straight_border_inner_corner_to_point.x > 0.0 ||
    straight_border_inner_corner_to_point.y > 0.0;


  // Whether the point is far enough inside the quad, such that the pixels are
  // not affected by the straight border.
  bool is_within_inner_straight_border =
    straight_border_inner_corner_to_point.x < -antialias_threshold &&
    straight_border_inner_corner_to_point.y < -antialias_threshold;

  // Fast path for points that must be part of the background. The quad's own
  // outline cannot reach this far in, but the content mask still can: this is
  // the interior of every quad that has a border or corners of its own, which
  // is most of them.
  if (is_within_inner_straight_border && !is_near_rounded_corner) {
    background_color.a *= mask_alpha;
    return background_color;
  }

  // Signed distance of the point to the outside edge of the quad's border
  float outer_sdf = quad_sdf_impl(corner_center_to_point, corner_radius);

  // Approximate signed distance of the point to the inside edge of the quad's
  // border. It is negative outside this edge (within the border), and
  // positive inside.
  //
  // This is not always an accurate signed distance:
  // * The rounded portions with varying border width use an approximation of
  //   nearest-point-on-ellipse.
  // * When it is quickly known to be outside the edge, -1.0 is used.
  float inner_sdf = 0.0;
  if (corner_center_to_point.x <= 0.0 || corner_center_to_point.y <= 0.0) {
    // Fast paths for straight borders
    inner_sdf = -max(straight_border_inner_corner_to_point.x,
                     straight_border_inner_corner_to_point.y);
  } else if (is_beyond_inner_straight_border) {
    // Fast path for points that must be outside the inner edge
    inner_sdf = -1.0;
  } else if (reduced_border.x == reduced_border.y) {
    // Fast path for circular inner edge.
    inner_sdf = -(outer_sdf + reduced_border.x);
  } else {
    float2 ellipse_radii = max(float2(0.0), float2(corner_radius) - reduced_border);
    inner_sdf = quarter_ellipse_sdf(corner_center_to_point, ellipse_radii);
  }

  // Negative when inside the border
  float border_sdf = max(inner_sdf, outer_sdf);

  float4 color = background_color;
  if (border_sdf < antialias_threshold) {
    float4 border_color = input.border_color;

    // Dashed border logic when border_style == 1
    if (quad.border_style == 1) {
      // Position along the perimeter in "dash space", where each dash
      // period has length 1
      float t = 0.0;

      // Total number of dash periods, so that the dash spacing can be
      // adjusted to evenly divide it
      float max_t = 0.0;

      // Border width is proportional to dash size. This is the behavior
      // used by browsers, but also avoids dashes from different segments
      // overlapping when dash size is smaller than the border width.
      //
      // Dash pattern: (2 * border width) dash, (1 * border width) gap
      const float dash_length_per_width = 2.0;
      const float dash_gap_per_width = 1.0;
      const float dash_period_per_width = dash_length_per_width + dash_gap_per_width;

      // Since the dash size is determined by border width, the density of
      // dashes varies. Multiplying a pixel distance by this returns a
      // position in dash space - it has units (dash period / pixels). So
      // a dash velocity of (1 / 10) is 1 dash every 10 pixels.
      float dash_velocity = 0.0;

      // Dividing this by the border width gives the dash velocity
      const float dv_numerator = 1.0 / dash_period_per_width;

      if (unrounded) {
        // When corners aren't rounded, the dashes are separately laid
        // out on each straight line, rather than around the whole
        // perimeter. This way each line starts and ends with a dash.
        bool is_horizontal = corner_center_to_point.x < corner_center_to_point.y;

        // Choosing the right border width for dashed borders.
        // TODO: A better solution exists taking a look at the whole file.
        // this does not fix single dashed borders at the corners
        float2 dashed_border = float2(
        fmax(quad.border_widths.bottom, quad.border_widths.top),
        fmax(quad.border_widths.right, quad.border_widths.left));

        float border_width = is_horizontal ? dashed_border.x : dashed_border.y;
        dash_velocity = dv_numerator / border_width;
        t = is_horizontal ? point.x : point.y;
        t *= dash_velocity;
        max_t = is_horizontal ? size.x : size.y;
        max_t *= dash_velocity;
      } else {
        // When corners are rounded, the dashes are laid out clockwise
        // around the whole perimeter.

        float r_tr = quad.corner_radii.top_right;
        float r_br = quad.corner_radii.bottom_right;
        float r_bl = quad.corner_radii.bottom_left;
        float r_tl = quad.corner_radii.top_left;

        float w_t = quad.border_widths.top;
        float w_r = quad.border_widths.right;
        float w_b = quad.border_widths.bottom;
        float w_l = quad.border_widths.left;

        // Straight side dash velocities
        float dv_t = w_t <= 0.0 ? 0.0 : dv_numerator / w_t;
        float dv_r = w_r <= 0.0 ? 0.0 : dv_numerator / w_r;
        float dv_b = w_b <= 0.0 ? 0.0 : dv_numerator / w_b;
        float dv_l = w_l <= 0.0 ? 0.0 : dv_numerator / w_l;

        // Straight side lengths in dash space
        float s_t = (size.x - r_tl - r_tr) * dv_t;
        float s_r = (size.y - r_tr - r_br) * dv_r;
        float s_b = (size.x - r_br - r_bl) * dv_b;
        float s_l = (size.y - r_bl - r_tl) * dv_l;

        float corner_dash_velocity_tr = corner_dash_velocity(dv_t, dv_r);
        float corner_dash_velocity_br = corner_dash_velocity(dv_b, dv_r);
        float corner_dash_velocity_bl = corner_dash_velocity(dv_b, dv_l);
        float corner_dash_velocity_tl = corner_dash_velocity(dv_t, dv_l);

        // Corner lengths in dash space
        float c_tr = r_tr * (M_PI_F / 2.0) * corner_dash_velocity_tr;
        float c_br = r_br * (M_PI_F / 2.0) * corner_dash_velocity_br;
        float c_bl = r_bl * (M_PI_F / 2.0) * corner_dash_velocity_bl;
        float c_tl = r_tl * (M_PI_F / 2.0) * corner_dash_velocity_tl;

        // Cumulative dash space upto each segment
        float upto_tr = s_t;
        float upto_r = upto_tr + c_tr;
        float upto_br = upto_r + s_r;
        float upto_b = upto_br + c_br;
        float upto_bl = upto_b + s_b;
        float upto_l = upto_bl + c_bl;
        float upto_tl = upto_l + s_l;
        max_t = upto_tl + c_tl;

        if (is_near_rounded_corner) {
          float radians = atan2(corner_center_to_point.y, corner_center_to_point.x);
          float corner_t = radians * corner_radius;

          if (center_to_point.x >= 0.0) {
            if (center_to_point.y < 0.0) {
              dash_velocity = corner_dash_velocity_tr;
              // Subtracted because radians is pi/2 to 0 when
              // going clockwise around the top right corner,
              // since the y axis has been flipped
              t = upto_r - corner_t * dash_velocity;
            } else {
              dash_velocity = corner_dash_velocity_br;
              // Added because radians is 0 to pi/2 when going
              // clockwise around the bottom-right corner
              t = upto_br + corner_t * dash_velocity;
            }
          } else {
            if (center_to_point.y >= 0.0) {
              dash_velocity = corner_dash_velocity_bl;
              // Subtracted because radians is pi/1 to 0 when
              // going clockwise around the bottom-left corner,
              // since the x axis has been flipped
              t = upto_l - corner_t * dash_velocity;
            } else {
              dash_velocity = corner_dash_velocity_tl;
              // Added because radians is 0 to pi/2 when going
              // clockwise around the top-left corner, since both
              // axis were flipped
              t = upto_tl + corner_t * dash_velocity;
            }
          }
        } else {
          // Straight borders
          bool is_horizontal = corner_center_to_point.x < corner_center_to_point.y;
          if (is_horizontal) {
            if (center_to_point.y < 0.0) {
              dash_velocity = dv_t;
              t = (point.x - r_tl) * dash_velocity;
            } else {
              dash_velocity = dv_b;
              t = upto_bl - (point.x - r_bl) * dash_velocity;
            }
          } else {
            if (center_to_point.x < 0.0) {
              dash_velocity = dv_l;
              t = upto_tl - (point.y - r_tl) * dash_velocity;
            } else {
              dash_velocity = dv_r;
              t = upto_r + (point.y - r_tr) * dash_velocity;
            }
          }
        }
      }

      float dash_length = dash_length_per_width / dash_period_per_width;
      float desired_dash_gap = dash_gap_per_width / dash_period_per_width;

      // Straight borders should start and end with a dash, so max_t is
      // reduced to cause this.
      max_t -= unrounded ? dash_length : 0.0;
      if (max_t >= 1.0) {
        // Adjust dash gap to evenly divide max_t
        float dash_count = floor(max_t);
        float dash_period = max_t / dash_count;
        border_color.a *= dash_alpha(t, dash_period, dash_length, dash_velocity,
                                     antialias_threshold);
      } else if (unrounded) {
        // When there isn't enough space for the full gap between the
        // two start / end dashes of a straight border, reduce gap to
        // make them fit.
        float dash_gap = max_t - dash_length;
        if (dash_gap > 0.0) {
          float dash_period = dash_length + dash_gap;
          border_color.a *= dash_alpha(t, dash_period, dash_length, dash_velocity,
                                       antialias_threshold);
        }
      }
    }

    // Blend the border on top of the background and then linearly interpolate
    // between the two as we slide inside the background.
    float4 blended_border = over(background_color, border_color);
    color = mix(background_color, blended_border,
                saturate(antialias_threshold - inner_sdf));
  }

  return color *
         float4(1.0, 1.0, 1.0,
                saturate(antialias_threshold - outer_sdf) * mask_alpha);
}

// Returns the dash velocity of a corner given the dash velocity of the two
// sides, by returning the slower velocity (larger dashes).
//
// Since 0 is used for dash velocity when the border width is 0 (instead of
// +inf), this returns the other dash velocity in that case.
//
// An alternative to this might be to appropriately interpolate the dash
// velocity around the corner, but that seems overcomplicated.
float corner_dash_velocity(float dv1, float dv2) {
  if (dv1 == 0.0) {
    return dv2;
  } else if (dv2 == 0.0) {
    return dv1;
  } else {
    return min(dv1, dv2);
  }
}

// Returns alpha used to render antialiased dashes.
// `t` is within the dash when `fmod(t, period) < length`.
float dash_alpha(
    float t, float period, float length, float dash_velocity,
    float antialias_threshold) {
  float half_period = period / 2.0;
  float half_length = length / 2.0;
  // Value in [-half_period, half_period]
  // The dash is in [-half_length, half_length]
  float centered = fmod(t + half_period - half_length, period) - half_period;
  // Signed distance for the dash, negative values are inside the dash
  float signed_distance = abs(centered) - half_length;
  // Antialiased alpha based on the signed distance
  return saturate(antialias_threshold - signed_distance / dash_velocity);
}

// This approximates distance to the nearest point to a quarter ellipse in a way
// that is sufficient for anti-aliasing when the ellipse is not very eccentric.
// The components of `point` are expected to be positive.
//
// Negative on the outside and positive on the inside.
float quarter_ellipse_sdf(float2 point, float2 radii) {
  // Scale the space to treat the ellipse like a unit circle
  float2 circle_vec = point / radii;
  float unit_circle_sdf = length(circle_vec) - 1.0;
  // Approximate up-scaling of the length by using the average of the radii.
  //
  // TODO: A better solution would be to use the gradient of the implicit
  // function for an ellipse to approximate a scaling factor.
  return unit_circle_sdf * (radii.x + radii.y) * -0.5;
}

struct ShadowVertexOutput {
  float4 position [[position]];
  float2 scene_position;
  float4 color [[flat]];
  uint shadow_id [[flat]];
  float clip_distance [[clip_distance]][4];
};

struct ShadowFragmentInput {
  float4 position [[position]];
  float2 scene_position;
  float4 color [[flat]];
  uint shadow_id [[flat]];
};

vertex ShadowVertexOutput shadow_vertex(
    uint unit_vertex_id [[vertex_id]], uint shadow_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(ShadowInputIndex_Vertices)]],
    constant Shadow *shadows [[buffer(ShadowInputIndex_Shadows)]],
    constant RenderTarget *render_target
    [[buffer(ShadowInputIndex_RenderTarget)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  Shadow shadow = shadows[shadow_id];

  Bounds_ScaledPixels bounds;
  if (shadow.inset != 0u) {
    bounds = shadow.element_bounds;
  } else {
    // Leave room for the gaussian tail outside the shadow rect.
    float margin = 3. * shadow.blur_radius;
    bounds = shadow.bounds;
    bounds.origin.x -= margin;
    bounds.origin.y -= margin;
    bounds.size.width += 2. * margin;
    bounds.size.height += 2. * margin;
  }

  float2 scene_position = to_scene_position(unit_vertex, bounds);
  float4 device_position = to_target_position(scene_position, render_target);
  float4 clip_distance =
      distance_from_clip_rect(unit_vertex, bounds, shadow.content_mask.bounds);
  float4 color = hsla_to_rgba(shadow.color);

  return ShadowVertexOutput{
      device_position,
      scene_position,
      color,
      shadow_id,
      {clip_distance.x, clip_distance.y, clip_distance.z, clip_distance.w}};
}

fragment float4 shadow_fragment(ShadowFragmentInput input [[stage_in]],
                                constant Shadow *shadows
                                [[buffer(ShadowInputIndex_Shadows)]],
                                constant ClipMask *clip_masks
                                [[buffer(ShadowInputIndex_ClipMasks),
                                  function_constant(CLIPPED)]],
                                texture2d<float> clip_atlas
                                [[texture(ShadowInputIndex_ClipAtlas),
                                  function_constant(CLIPPED)]]) {
  Shadow shadow = shadows[input.shadow_id];

  float2 origin = float2(shadow.bounds.origin.x, shadow.bounds.origin.y);
  float2 size = float2(shadow.bounds.size.width, shadow.bounds.size.height);
  float2 half_size = size / 2.;
  float2 center = origin + half_size;
  float2 point = input.scene_position - center;
  float corner_radius;
  if (point.x < 0.) {
    if (point.y < 0.) {
      corner_radius = shadow.corner_radii.top_left;
    } else {
      corner_radius = shadow.corner_radii.bottom_left;
    }
  } else {
    if (point.y < 0.) {
      corner_radius = shadow.corner_radii.top_right;
    } else {
      corner_radius = shadow.corner_radii.bottom_right;
    }
  }

  float alpha;
  if (shadow.blur_radius == 0.) {
    float distance = quad_sdf(input.scene_position, shadow.bounds, shadow.corner_radii);
    alpha = saturate(0.5 - distance);
  } else {
    // The signal is only non-zero in a limited range, so don't waste samples
    float low = point.y - half_size.y;
    float high = point.y + half_size.y;
    float start = clamp(-3. * shadow.blur_radius, low, high);
    float end = clamp(3. * shadow.blur_radius, low, high);

    // Accumulate samples (we can get away with surprisingly few samples)
    float step = (end - start) / 4.;
    float y = start + step * 0.5;
    alpha = 0.;
    for (int i = 0; i < 4; i++) {
      alpha += blur_along_x(point.x, point.y - y, shadow.blur_radius,
                            corner_radius, half_size) *
               gaussian(y, shadow.blur_radius) * step;
      y += step;
    }
  }

  if (shadow.inset != 0u) {
    // The inset shadow is the complement of the (blurred) hole rect, clipped to the element.
    // `saturate(0.5 - d)` gives a 1-pixel antialiased edge: d <= -0.5 -> 1, d >= 0.5 -> 0.
    alpha = 1. - alpha;
    float element_distance = quad_sdf(input.scene_position, shadow.element_bounds,
                                      shadow.element_corner_radii);
    alpha *= saturate(0.5 - element_distance);
  }

  alpha *= content_mask_alpha(input.scene_position, shadow.content_mask);
  if (CLIPPED) {
    alpha *=
        clip_mask_alpha(input.scene_position, shadow.clip, clip_masks, clip_atlas);
  }

  return input.color * float4(1., 1., 1., alpha);
}

struct UnderlineVertexOutput {
  float4 position [[position]];
  float2 scene_position;
  float4 color [[flat]];
  uint underline_id [[flat]];
  float clip_distance [[clip_distance]][4];
};

struct UnderlineFragmentInput {
  float4 position [[position]];
  float2 scene_position;
  float4 color [[flat]];
  uint underline_id [[flat]];
};

vertex UnderlineVertexOutput underline_vertex(
    uint unit_vertex_id [[vertex_id]], uint underline_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(UnderlineInputIndex_Vertices)]],
    constant Underline *underlines [[buffer(UnderlineInputIndex_Underlines)]],
    constant RenderTarget *render_target
    [[buffer(UnderlineInputIndex_RenderTarget)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  Underline underline = underlines[underline_id];
  float2 scene_position = to_scene_position(unit_vertex, underline.bounds);
  float4 device_position = to_target_position(scene_position, render_target);
  float4 clip_distance = distance_from_clip_rect(unit_vertex, underline.bounds,
                                                 underline.content_mask.bounds);
  float4 color = hsla_to_rgba(underline.color);
  return UnderlineVertexOutput{
      device_position,
      scene_position,
      color,
      underline_id,
      {clip_distance.x, clip_distance.y, clip_distance.z, clip_distance.w}};
}

fragment float4 underline_fragment(UnderlineFragmentInput input [[stage_in]],
                                   constant Underline *underlines
                                   [[buffer(UnderlineInputIndex_Underlines)]],
                                   constant ClipMask *clip_masks
                                   [[buffer(UnderlineInputIndex_ClipMasks),
                                     function_constant(CLIPPED)]],
                                   texture2d<float> clip_atlas
                                   [[texture(UnderlineInputIndex_ClipAtlas),
                                     function_constant(CLIPPED)]]) {
  const float WAVE_FREQUENCY = 2.0;
  const float WAVE_HEIGHT_RATIO = 0.8;

  Underline underline = underlines[input.underline_id];
  // Straight alpha out of this pipeline, so the mask multiplies the alpha
  // channel alone.
  float mask_alpha =
      content_mask_alpha(input.scene_position, underline.content_mask);
  if (CLIPPED) {
    mask_alpha *= clip_mask_alpha(input.scene_position, underline.clip, clip_masks,
                                  clip_atlas);
  }
  if (underline.wavy) {
    float half_thickness = underline.thickness * 0.5;
    float2 origin =
        float2(underline.bounds.origin.x, underline.bounds.origin.y);

    float2 st = ((input.scene_position - origin) / underline.bounds.size.height) -
                float2(0., 0.5);
    float frequency = (M_PI_F * WAVE_FREQUENCY * underline.thickness) / underline.bounds.size.height;
    float amplitude = (underline.thickness * WAVE_HEIGHT_RATIO) / underline.bounds.size.height;

    float sine = sin(st.x * frequency) * amplitude;
    float dSine = cos(st.x * frequency) * amplitude * frequency;
    float distance = (st.y - sine) / sqrt(1. + dSine * dSine);
    float distance_in_pixels = distance * underline.bounds.size.height;
    float distance_from_top_border = distance_in_pixels - half_thickness;
    float distance_from_bottom_border = distance_in_pixels + half_thickness;
    float alpha = saturate(
        0.5 - max(-distance_from_bottom_border, distance_from_top_border));
    return input.color * float4(1., 1., 1., alpha * mask_alpha);
  } else {
    return input.color * float4(1., 1., 1., mask_alpha);
  }
}

// Every glyph in the app goes through this pipeline, so the fragment stage is
// given the mask as two flat varyings rather than the sprite's index: reading
// the sprite back out of the instance buffer would load the whole struct once
// per fragment, for data the vertex stage has already loaded.
struct MonochromeSpriteVertexOutput {
  float4 position [[position]];
  float2 scene_position;
  float2 tile_position;
  float4 color [[flat]];
  float4 mask_bounds [[flat]];
  float4 mask_corner_radii [[flat]];
  float4 clip_distance;
  // Carried only by the clipped variant, for the same reason the mask above is
  // carried as varyings at all: the fragment stage never loads the sprite.
  uint clip_id [[flat, function_constant(CLIPPED)]];
};

struct MonochromeSpriteFragmentInput {
  float4 position [[position]];
  float2 scene_position;
  float2 tile_position;
  float4 color [[flat]];
  float4 mask_bounds [[flat]];
  float4 mask_corner_radii [[flat]];
  float4 clip_distance;
  uint clip_id [[flat, function_constant(CLIPPED)]];
};

vertex MonochromeSpriteVertexOutput monochrome_sprite_vertex(
    uint unit_vertex_id [[vertex_id]], uint sprite_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(SpriteInputIndex_Vertices)]],
    constant MonochromeSprite *sprites [[buffer(SpriteInputIndex_Sprites)]],
    constant RenderTarget *render_target
    [[buffer(SpriteInputIndex_RenderTarget)]],
    constant Size_DevicePixels *atlas_size
    [[buffer(SpriteInputIndex_AtlasTextureSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  MonochromeSprite sprite = sprites[sprite_id];
  float2 scene_position = to_scene_position_transformed(
      unit_vertex, sprite.bounds, sprite.transformation);
  float4 device_position = to_target_position(scene_position, render_target);
  float4 clip_distance = distance_from_clip_rect_transformed(unit_vertex, sprite.bounds,
                                                 sprite.content_mask.bounds, sprite.transformation);
  float2 tile_position = to_tile_position(unit_vertex, sprite.tile, atlas_size);
  float4 color = hsla_to_rgba(sprite.color);
  MonochromeSpriteVertexOutput output;
  output.position = device_position;
  output.scene_position = scene_position;
  output.tile_position = tile_position;
  output.color = color;
  output.mask_bounds = pack_bounds(sprite.content_mask.bounds);
  output.mask_corner_radii = pack_corner_radii(sprite.content_mask.corner_radii);
  output.clip_distance = clip_distance;
  if (CLIPPED) {
    output.clip_id = sprite.clip;
  }
  return output;
}

fragment float4 monochrome_sprite_fragment(
    MonochromeSpriteFragmentInput input [[stage_in]],
    texture2d<float> atlas_texture [[texture(SpriteInputIndex_AtlasTexture)]],
    constant ClipMask *clip_masks
    [[buffer(SpriteInputIndex_ClipMasks), function_constant(CLIPPED)]],
    texture2d<float> clip_atlas
    [[texture(SpriteInputIndex_ClipAtlas), function_constant(CLIPPED)]]) {
  if (any(input.clip_distance < float4(0.0))) {
    return float4(0.0);
  }

  constexpr sampler atlas_texture_sampler(mag_filter::linear,
                                          min_filter::linear);
  float4 sample =
      atlas_texture.sample(atlas_texture_sampler, input.tile_position);
  float4 color = input.color;
  color.a *= sample.a * packed_content_mask_alpha(input.scene_position,
                                                  input.mask_bounds,
                                                  input.mask_corner_radii);
  if (CLIPPED) {
    color.a *= clip_mask_alpha(input.scene_position, input.clip_id, clip_masks,
                               clip_atlas);
  }
  return color;
}

struct PolychromeSpriteVertexOutput {
  float4 position [[position]];
  float2 scene_position;
  float2 tile_position;
  uint sprite_id [[flat]];
  float clip_distance [[clip_distance]][4];
};

struct PolychromeSpriteFragmentInput {
  float4 position [[position]];
  float2 scene_position;
  float2 tile_position;
  uint sprite_id [[flat]];
};

vertex PolychromeSpriteVertexOutput polychrome_sprite_vertex(
    uint unit_vertex_id [[vertex_id]], uint sprite_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(SpriteInputIndex_Vertices)]],
    constant PolychromeSprite *sprites [[buffer(SpriteInputIndex_Sprites)]],
    constant RenderTarget *render_target
    [[buffer(SpriteInputIndex_RenderTarget)]],
    constant Size_DevicePixels *atlas_size
    [[buffer(SpriteInputIndex_AtlasTextureSize)]]) {

  float2 unit_vertex = unit_vertices[unit_vertex_id];
  PolychromeSprite sprite = sprites[sprite_id];
  float2 scene_position = to_scene_position(unit_vertex, sprite.bounds);
  float4 device_position = to_target_position(scene_position, render_target);
  float4 clip_distance = distance_from_clip_rect(unit_vertex, sprite.bounds,
                                                 sprite.content_mask.bounds);
  float2 tile_position = to_tile_position(unit_vertex, sprite.tile, atlas_size);
  return PolychromeSpriteVertexOutput{
      device_position,
      scene_position,
      tile_position,
      sprite_id,
      {clip_distance.x, clip_distance.y, clip_distance.z, clip_distance.w}};
}

fragment float4 polychrome_sprite_fragment(
    PolychromeSpriteFragmentInput input [[stage_in]],
    constant PolychromeSprite *sprites [[buffer(SpriteInputIndex_Sprites)]],
    texture2d<float> atlas_texture [[texture(SpriteInputIndex_AtlasTexture)]],
    constant ClipMask *clip_masks
    [[buffer(SpriteInputIndex_ClipMasks), function_constant(CLIPPED)]],
    texture2d<float> clip_atlas
    [[texture(SpriteInputIndex_ClipAtlas), function_constant(CLIPPED)]]) {
  PolychromeSprite sprite = sprites[input.sprite_id];
  constexpr sampler atlas_texture_sampler(mag_filter::linear,
                                          min_filter::linear);
  float4 sample =
      atlas_texture.sample(atlas_texture_sampler, input.tile_position);
  float distance =
      quad_sdf(input.scene_position, sprite.bounds, sprite.corner_radii);

  float4 color = sample;
  if (sprite.grayscale) {
    float grayscale = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
    color.r = grayscale;
    color.g = grayscale;
    color.b = grayscale;
  }
  color.a *= sprite.opacity * saturate(0.5 - distance) *
             content_mask_alpha(input.scene_position, sprite.content_mask);
  if (CLIPPED) {
    color.a *=
        clip_mask_alpha(input.scene_position, sprite.clip, clip_masks, clip_atlas);
  }
  return color;
}

struct PathRasterizationVertexOutput {
  float4 position [[position]];
  float2 st_position;
  uint vertex_id [[flat]];
  float clip_rect_distance [[clip_distance]][4];
};

struct PathRasterizationFragmentInput {
  float4 position [[position]];
  float2 st_position;
  uint vertex_id [[flat]];
};

vertex PathRasterizationVertexOutput path_rasterization_vertex(
  uint vertex_id [[vertex_id]],
  constant PathRasterizationVertex *vertices [[buffer(PathRasterizationInputIndex_Vertices)]],
  constant Size_DevicePixels *atlas_size [[buffer(PathRasterizationInputIndex_ViewportSize)]]
) {
  PathRasterizationVertex v = vertices[vertex_id];
  float2 vertex_position = float2(v.xy_position.x, v.xy_position.y);
  float4 position = float4(
    vertex_position * float2(2. / atlas_size->width, -2. / atlas_size->height) + float2(-1., 1.),
    0.,
    1.
  );
  return PathRasterizationVertexOutput{
      position,
      float2(v.st_position.x, v.st_position.y),
      vertex_id,
      {
        v.xy_position.x - v.bounds.origin.x,
        v.bounds.origin.x + v.bounds.size.width - v.xy_position.x,
        v.xy_position.y - v.bounds.origin.y,
        v.bounds.origin.y + v.bounds.size.height - v.xy_position.y
      }
  };
}

fragment float4 path_rasterization_fragment(
  PathRasterizationFragmentInput input [[stage_in]],
  constant PathRasterizationVertex *vertices [[buffer(PathRasterizationInputIndex_Vertices)]],
  constant ContentMask_ScaledPixels *content_masks [[buffer(PathRasterizationInputIndex_ContentMasks)]],
  constant PathBrushRecord *path_brushes [[buffer(PathRasterizationInputIndex_Brushes)]],
  constant uint *path_brush_count [[buffer(PathRasterizationInputIndex_BrushCount)]],
  texture2d<float> brush_atlas [[texture(PathRasterizationInputIndex_BrushAtlas)]],
  constant uint *path_clips [[buffer(PathRasterizationInputIndex_ClipIds), function_constant(CLIPPED)]],
  constant ClipMask *clip_masks [[buffer(PathRasterizationInputIndex_ClipMasks), function_constant(CLIPPED)]],
  texture2d<float> clip_atlas [[texture(PathRasterizationInputIndex_ClipAtlas), function_constant(CLIPPED)]]
) {
  float2 dx = dfdx(input.st_position);
  float2 dy = dfdy(input.st_position);

  PathRasterizationVertex v = vertices[input.vertex_id];
  Bounds_ScaledPixels path_bounds = v.bounds;
  float alpha;
  if (length(float2(dx.x, dy.x)) < 0.001) {
    alpha = 1.0;
  } else {
    float2 gradient = float2(
      (2. * input.st_position.x) * dx.x - dx.y,
      (2. * input.st_position.x) * dy.x - dy.y
    );
    float f = (input.st_position.x * input.st_position.x) - input.st_position.y;
    float distance = f / length(gradient);
    alpha = saturate(0.5 - distance);
  }

  alpha *= content_mask_alpha(input.position.xy, content_masks[v.path_id]);
  // A path takes its clip here, while it is being rasterized into the
  // intermediate, exactly the way it takes its content mask: the sprite pass
  // that copies the intermediate out applies neither.
  if (CLIPPED) {
    alpha *= clip_mask_alpha(input.position.xy, path_clips[v.path_id],
                             clip_masks, clip_atlas);
  }

  // An image brush replaces the fill entirely, and is resolved here rather than
  // in the pass that copies the intermediate out: that pass copies a union rect
  // whenever the paths in a batch disagree on their draw order, by which point
  // several paths' coverages have already been blended into one pixel and there
  // is no path left to ask which image it wanted.
  if (v.path_id < *path_brush_count) {
    PathBrushRecord record = path_brushes[v.path_id];
    if (record.enabled != 0) {
      float4 sample = sample_path_brush(record.brush, input.position.xy, brush_atlas);
      // The atlas holds straight (un-premultiplied) BGRA, and this pipeline
      // blends One / OneMinusSourceAlpha, so what leaves here is premultiplied.
      float brush_alpha = sample.a * record.brush.opacity * alpha;
      return float4(sample.rgb * brush_alpha, brush_alpha);
    }
  }

  Background background = v.color;
  GradientColor gradient_color = prepare_fill_color(
    background.tag,
    background.color_space,
    background.solid,
    background.colors[0].color,
    background.colors[1].color
  );

  float4 color = fill_color(
    background,
    input.position.xy,
    path_bounds,
    gradient_color.solid,
    gradient_color.color0,
    gradient_color.color1
  );

  return float4(color.rgb * color.a * alpha, alpha * color.a);
}

struct PathSpriteVertexOutput {
  float4 position [[position]];
  float2 texture_coords;
};

vertex PathSpriteVertexOutput path_sprite_vertex(
  uint unit_vertex_id [[vertex_id]],
  uint sprite_id [[instance_id]],
  constant float2 *unit_vertices [[buffer(SpriteInputIndex_Vertices)]],
  constant PathSprite *sprites [[buffer(SpriteInputIndex_Sprites)]],
  constant RenderTarget *render_target [[buffer(SpriteInputIndex_RenderTarget)]],
  // The intermediate a path was rasterized into is the whole window, whatever
  // target this copy is landing on, so its size is bound separately from the
  // target's.
  constant Size_DevicePixels *intermediate_size [[buffer(SpriteInputIndex_AtlasTextureSize)]]
) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  PathSprite sprite = sprites[sprite_id];
  // Don't apply content mask because it was already accounted for when
  // rasterizing the path.
  float2 scene_position = to_scene_position(unit_vertex, sprite.bounds);
  float4 device_position = to_target_position(scene_position, render_target);

  float2 texture_coords = scene_position / float2(intermediate_size->width, intermediate_size->height);

  return PathSpriteVertexOutput{
    device_position,
    texture_coords
  };
}

fragment float4 path_sprite_fragment(
  PathSpriteVertexOutput input [[stage_in]],
  texture2d<float> intermediate_texture [[texture(SpriteInputIndex_AtlasTexture)]]
) {
  constexpr sampler intermediate_texture_sampler(mag_filter::linear, min_filter::linear);
  return intermediate_texture.sample(intermediate_texture_sampler, input.texture_coords);
}

struct SurfaceVertexOutput {
  float4 position [[position]];
  float2 scene_position;
  float2 texture_position;
  float4 mask_bounds [[flat]];
  float4 mask_corner_radii [[flat]];
  float clip_distance [[clip_distance]][4];
  uint clip_id [[flat, function_constant(CLIPPED)]];
};

struct SurfaceFragmentInput {
  float4 position [[position]];
  float2 scene_position;
  float2 texture_position;
  float4 mask_bounds [[flat]];
  float4 mask_corner_radii [[flat]];
  uint clip_id [[flat, function_constant(CLIPPED)]];
};

vertex SurfaceVertexOutput surface_vertex(
    uint unit_vertex_id [[vertex_id]], uint surface_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(SurfaceInputIndex_Vertices)]],
    constant SurfaceBounds *surfaces [[buffer(SurfaceInputIndex_Surfaces)]],
    constant RenderTarget *render_target
    [[buffer(SurfaceInputIndex_RenderTarget)]],
    constant Size_DevicePixels *texture_size
    [[buffer(SurfaceInputIndex_TextureSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  SurfaceBounds surface = surfaces[surface_id];
  float2 scene_position = to_scene_position(unit_vertex, surface.bounds);
  float4 device_position = to_target_position(scene_position, render_target);
  float4 clip_distance = distance_from_clip_rect(unit_vertex, surface.bounds,
                                                 surface.content_mask.bounds);
  // We are going to copy the whole texture, so the texture position corresponds
  // to the current vertex of the unit triangle.
  float2 texture_position = unit_vertex;
  SurfaceVertexOutput output;
  output.position = device_position;
  output.scene_position = scene_position;
  output.texture_position = texture_position;
  output.mask_bounds = pack_bounds(surface.content_mask.bounds);
  output.mask_corner_radii = pack_corner_radii(surface.content_mask.corner_radii);
  output.clip_distance[0] = clip_distance.x;
  output.clip_distance[1] = clip_distance.y;
  output.clip_distance[2] = clip_distance.z;
  output.clip_distance[3] = clip_distance.w;
  if (CLIPPED) {
    output.clip_id = surface.clip;
  }
  return output;
}

fragment float4 surface_fragment(SurfaceFragmentInput input [[stage_in]],
                                 texture2d<float> y_texture
                                 [[texture(SurfaceInputIndex_YTexture)]],
                                 texture2d<float> cb_cr_texture
                                 [[texture(SurfaceInputIndex_CbCrTexture)]],
                                 constant ClipMask *clip_masks
                                 [[buffer(SurfaceInputIndex_ClipMasks),
                                   function_constant(CLIPPED)]],
                                 texture2d<float> clip_atlas
                                 [[texture(SurfaceInputIndex_ClipAtlas),
                                   function_constant(CLIPPED)]]) {
  constexpr sampler texture_sampler(mag_filter::linear, min_filter::linear);
  const float4x4 ycbcrToRGBTransform =
      float4x4(float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
               float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f),
               float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
               float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f));
  float4 ycbcr = float4(
      y_texture.sample(texture_sampler, input.texture_position).r,
      cb_cr_texture.sample(texture_sampler, input.texture_position).rg, 1.0);

  float4 color = ycbcrToRGBTransform * ycbcr;
  // The `surfaces` pipeline blends straight alpha, so only the alpha channel
  // takes the mask.
  color.a *= packed_content_mask_alpha(input.scene_position, input.mask_bounds,
                                       input.mask_corner_radii);
  if (CLIPPED) {
    color.a *= clip_mask_alpha(input.scene_position, input.clip_id, clip_masks,
                               clip_atlas);
  }
  return color;
}

fragment float4 surface_bgra_fragment(SurfaceFragmentInput input [[stage_in]],
                                      texture2d<float> bgra_texture
                                      [[texture(SurfaceInputIndex_YTexture)]],
                                      constant ClipMask *clip_masks
                                      [[buffer(SurfaceInputIndex_ClipMasks),
                                        function_constant(CLIPPED)]],
                                      texture2d<float> clip_atlas
                                      [[texture(SurfaceInputIndex_ClipAtlas),
                                        function_constant(CLIPPED)]]) {
  constexpr sampler texture_sampler(mag_filter::linear, min_filter::linear);
  // Unlike `surfaces` above, the `bgra_surfaces` pipeline blends premultiplied
  // alpha, so the mask scales the whole colour. Masking the alpha alone would
  // leave the colour un-darkened and fringe the cut edge.
  float alpha = packed_content_mask_alpha(input.scene_position, input.mask_bounds,
                                          input.mask_corner_radii);
  if (CLIPPED) {
    alpha *= clip_mask_alpha(input.scene_position, input.clip_id, clip_masks,
                             clip_atlas);
  }
  return bgra_texture.sample(texture_sampler, input.texture_position) * alpha;
}

// The two halves of stencil-and-cover, which is how a clip path becomes a tile
// of the coverage atlas.
//
// The stencil half draws a triangle fan per contour with colour writes off,
// counting winding into the stencil buffer: a fan's interior edges are walked
// once each way and cancel exactly, so a non-convex or self-overlapping contour
// comes out right where a plain fan of blended triangles - which is all a
// `Path` can do - would double-paint or paint outside itself.
//
// The cover half then draws the tile rectangle wherever the stencil says the
// path covers it, samples the parent clip's tile so nesting intersects, and
// zeroes the stencil behind itself so the next clip in the pass starts clean.
// Coverage comes from multisampling: the stencil test is per sample, and the
// resolve averages them.

struct ClipMaskVertexOutput {
  float4 position [[position]];
};

vertex ClipMaskVertexOutput clip_stencil_vertex(
    uint vertex_id [[vertex_id]],
    constant PointF *vertices [[buffer(ClipMaskInputIndex_Vertices)]],
    constant Size_DevicePixels *target_size
    [[buffer(ClipMaskInputIndex_TargetSize)]]) {
  PointF vertex_position = vertices[vertex_id];
  float2 position = float2(vertex_position.x, vertex_position.y);
  return ClipMaskVertexOutput{
      float4(position * float2(2. / target_size->width,
                               -2. / target_size->height) +
                 float2(-1., 1.),
             0., 1.)};
}

// The colour this writes is thrown away - the stencil pipeline's write mask is
// empty - but a pipeline with a colour attachment wants a fragment stage that
// produces one.
fragment float clip_stencil_fragment() { return 0.; }

vertex ClipMaskVertexOutput clip_cover_vertex(
    uint unit_vertex_id [[vertex_id]],
    constant float2 *unit_vertices [[buffer(ClipMaskInputIndex_Vertices)]],
    constant ClipCover *cover [[buffer(ClipMaskInputIndex_Cover)]],
    constant Size_DevicePixels *target_size
    [[buffer(ClipMaskInputIndex_TargetSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  return ClipMaskVertexOutput{
      to_device_position(unit_vertex, cover->tile, target_size)};
}

fragment float clip_cover_fragment(
    ClipMaskVertexOutput input [[stage_in]],
    constant ClipCover *cover [[buffer(ClipMaskInputIndex_Cover)]],
    texture2d<float> clip_atlas [[texture(ClipMaskInputIndex_ClipAtlas)]]) {
  if (cover->has_parent == 0u) {
    return 1.;
  }
  float2 parent_position =
      input.position.xy +
      float2(cover->parent_offset.x, cover->parent_offset.y);
  return clip_atlas.read(uint2(parent_position)).r;
}

// Which Porter-Duff operator the composite pipeline this was specialized into
// blends with. It selects what the fragment stage *emits*; the operator's src
// and dst factors are the pipeline's blend state, and the two are written to
// agree. See `GroupComposeMode`.
constant uint COMPOSE_MODE [[function_constant(1)]];

struct GroupCompositeVertexOutput {
  float4 position [[position]];
  float2 scene_position;
  float2 texture_position;
};

vertex GroupCompositeVertexOutput group_composite_vertex(
    uint unit_vertex_id [[vertex_id]],
    constant float2 *unit_vertices
    [[buffer(GroupCompositeInputIndex_Vertices)]],
    constant GroupComposite *composite
    [[buffer(GroupCompositeInputIndex_Composite)]],
    constant RenderTarget *render_target
    [[buffer(GroupCompositeInputIndex_RenderTarget)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  float2 scene_position = to_scene_position(unit_vertex, composite->bounds);
  // The group's target is at least the group's size and may be larger - it is
  // pooled and quantized - so the region to sample is the group's own bounds
  // in the corner of it, not the whole texture.
  float2 texture_position =
      unit_vertex *
      float2(composite->bounds.size.width, composite->bounds.size.height) /
      float2((float)composite->texture_size.width,
             (float)composite->texture_size.height);
  return GroupCompositeVertexOutput{
      to_target_position(scene_position, render_target), scene_position,
      texture_position};
}

// A group's finished target, composited onto what is underneath it.
//
// The target holds premultiplied RGBA. The clip path in force where the group
// was pushed cannot be a blend factor - it varies per fragment - so its
// coverage `c`, with the group's opacity folded in, is folded into what this
// emits instead, and every operator below is written so that `c == 0` leaves
// the destination exactly as it found it.
fragment float4 group_composite_fragment(
    GroupCompositeVertexOutput input [[stage_in]],
    texture2d<float> group_texture
    [[texture(GroupCompositeInputIndex_GroupTexture)]],
    constant GroupComposite *composite
    [[buffer(GroupCompositeInputIndex_Composite)]],
    constant GroupFilter *filter [[buffer(GroupCompositeInputIndex_Filter)]],
    constant ClipMask *clip_masks
    [[buffer(GroupCompositeInputIndex_ClipMasks),
      function_constant(CLIPPED)]],
    texture2d<float> clip_atlas
    [[texture(GroupCompositeInputIndex_ClipAtlas),
      function_constant(CLIPPED)]]) {
  // Nearest: the composite quad is pixel-aligned with the target it was drawn
  // into, so every sample is a texel centre and filtering would only blur it.
  constexpr sampler group_sampler(mag_filter::nearest, min_filter::nearest);
  float4 source = apply_color_matrices(
      group_texture.sample(group_sampler, input.texture_position), filter);

  float coverage = composite->opacity;
  if (CLIPPED) {
    coverage *= clip_mask_alpha(input.scene_position, composite->clip,
                                clip_masks, clip_atlas);
  }

  switch (COMPOSE_MODE) {
  case GroupComposeMode_DestOut:
    // dst * (1 - c * S.a)
    return float4(0., 0., 0., coverage * source.a);
  case GroupComposeMode_DestIn:
    // dst * (1 - c * (1 - S.a)): at c == 0 the destination is multiplied by
    // one, which is what "outside the group's clip, nothing happened" means.
    return float4(0., 0., 0., 1. - coverage * (1. - source.a));
  case GroupComposeMode_Clear:
    // dst * (1 - c)
    return float4(0., 0., 0., coverage);
  default:
    // Src-over, xor, src-atop and plus all take the source scaled by coverage
    // and differ only in the pipeline's blend factors.
    return source * coverage;
  }
}

float4 hsla_to_rgba(Hsla hsla) {
  float h = hsla.h * 6.0; // Now, it's an angle but scaled in [0, 6) range
  float s = hsla.s;
  float l = hsla.l;
  float a = hsla.a;

  float c = (1.0 - fabs(2.0 * l - 1.0)) * s;
  float x = c * (1.0 - fabs(fmod(h, 2.0) - 1.0));
  float m = l - c / 2.0;

  float r = 0.0;
  float g = 0.0;
  float b = 0.0;

  if (h >= 0.0 && h < 1.0) {
    r = c;
    g = x;
    b = 0.0;
  } else if (h >= 1.0 && h < 2.0) {
    r = x;
    g = c;
    b = 0.0;
  } else if (h >= 2.0 && h < 3.0) {
    r = 0.0;
    g = c;
    b = x;
  } else if (h >= 3.0 && h < 4.0) {
    r = 0.0;
    g = x;
    b = c;
  } else if (h >= 4.0 && h < 5.0) {
    r = x;
    g = 0.0;
    b = c;
  } else {
    r = c;
    g = 0.0;
    b = x;
  }

  float4 rgba;
  rgba.x = (r + m);
  rgba.y = (g + m);
  rgba.z = (b + m);
  rgba.w = a;
  return rgba;
}

float3 srgb_to_linear(float3 color) {
  return pow(color, float3(2.2));
}

float3 linear_to_srgb(float3 color) {
  return pow(color, float3(1.0 / 2.2));
}

// Converts a sRGB color to the Oklab color space.
// Reference: https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
float4 srgb_to_oklab(float4 color) {
  // Convert non-linear sRGB to linear sRGB
  color = float4(srgb_to_linear(color.rgb), color.a);

  float l = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
  float m = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
  float s = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;

  float l_ = pow(l, 1.0/3.0);
  float m_ = pow(m, 1.0/3.0);
  float s_ = pow(s, 1.0/3.0);

  return float4(
   	0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
   	1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
   	0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
   	color.a
  );
}

// Converts an Oklab color to the sRGB color space.
float4 oklab_to_srgb(float4 color) {
  float l_ = color.r + 0.3963377774 * color.g + 0.2158037573 * color.b;
  float m_ = color.r - 0.1055613458 * color.g - 0.0638541728 * color.b;
  float s_ = color.r - 0.0894841775 * color.g - 1.2914855480 * color.b;

  float l = l_ * l_ * l_;
  float m = m_ * m_ * m_;
  float s = s_ * s_ * s_;

  float3 linear_rgb = float3(
   	4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
   	-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
   	-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
  );

  // Convert linear sRGB to non-linear sRGB
  return float4(linear_to_srgb(linear_rgb), color.a);
}

float4 to_device_position(float2 unit_vertex, Bounds_ScaledPixels bounds,
                          constant Size_DevicePixels *input_viewport_size) {
  float2 position =
      unit_vertex * float2(bounds.size.width, bounds.size.height) +
      float2(bounds.origin.x, bounds.origin.y);
  float2 viewport_size = float2((float)input_viewport_size->width,
                                (float)input_viewport_size->height);
  float2 device_position =
      position / viewport_size * float2(2., -2.) + float2(-1., 1.);
  return float4(device_position, 0., 1.);
}

// Where a primitive's unit vertex lands in the window, in device pixels.
//
// This is the space every primitive is expressed in and every fragment stage
// reasons in - content masks, clip masks, gradients and signed distance fields
// alike - and it is *not* the framebuffer's space once a group is being drawn
// to a target of its own, which is why each vertex stage carries this across
// to its fragment stage rather than letting it read `[[position]]`.
float2 to_scene_position(float2 unit_vertex, Bounds_ScaledPixels bounds) {
  return unit_vertex * float2(bounds.size.width, bounds.size.height) +
         float2(bounds.origin.x, bounds.origin.y);
}

float2 to_scene_position_transformed(float2 unit_vertex,
                                     Bounds_ScaledPixels bounds,
                                     TransformationMatrix transformation) {
  float2 position = to_scene_position(unit_vertex, bounds);

  // Apply the transformation matrix to the position via matrix multiplication.
  float2 transformed_position = float2(0, 0);
  transformed_position[0] = position[0] * transformation.rotation_scale[0][0] + position[1] * transformation.rotation_scale[0][1];
  transformed_position[1] = position[0] * transformation.rotation_scale[1][0] + position[1] * transformation.rotation_scale[1][1];

  // Add in the translation component of the transformation matrix.
  transformed_position[0] += transformation.translation[0];
  transformed_position[1] += transformation.translation[1];

  return transformed_position;
}

// A scene position placed on the attachment currently being drawn into, which
// is the window itself only when no group is open.
float4 to_target_position(float2 scene_position,
                          constant RenderTarget *render_target) {
  float2 size = float2((float)render_target->size.width,
                       (float)render_target->size.height);
  float2 origin = float2((float)render_target->origin.x,
                         (float)render_target->origin.y);
  float2 device_position =
      (scene_position - origin) / size * float2(2., -2.) + float2(-1., 1.);
  return float4(device_position, 0., 1.);
}

// A chain of CSS colour filters, applied to a group's premultiplied result.
//
// Colour matrices are defined on non-premultiplied colour, so this undoes the
// premultiplication, runs the chain, and puts it back. The chain is applied one
// matrix at a time with a clamp between stages rather than collapsed into a
// single product, because the spec clamps between filter primitives and the two
// are not the same function: `brightness(2) invert(1)` diverges above 0.5.
float4 apply_color_matrices(float4 premultiplied, constant GroupFilter *filter) {
  if (filter->matrix_count == 0u) {
    return premultiplied;
  }
  float alpha = premultiplied.a;
  float4 color =
      float4(alpha > 0. ? premultiplied.rgb / alpha : float3(0.), alpha);
  for (uint index = 0u; index < filter->matrix_count; index++) {
    constant float *m = filter->matrices + index * 20u;
    float4 result;
    result.r = m[0] * color.r + m[1] * color.g + m[2] * color.b +
               m[3] * color.a + m[4];
    result.g = m[5] * color.r + m[6] * color.g + m[7] * color.b +
               m[8] * color.a + m[9];
    result.b = m[10] * color.r + m[11] * color.g + m[12] * color.b +
               m[13] * color.a + m[14];
    result.a = m[15] * color.r + m[16] * color.g + m[17] * color.b +
               m[18] * color.a + m[19];
    color = saturate(result);
  }
  return float4(color.rgb * color.a, color.a);
}


float2 to_tile_position(float2 unit_vertex, AtlasTile tile,
                        constant Size_DevicePixels *atlas_size) {
  float2 tile_origin = float2(tile.bounds.origin.x, tile.bounds.origin.y);
  float2 tile_size = float2(tile.bounds.size.width, tile.bounds.size.height);
  return (tile_origin + unit_vertex * tile_size) /
         float2((float)atlas_size->width, (float)atlas_size->height);
}

// Folds a texel index, which an image brush may take anywhere on the number
// line, back onto the [0, extent) the tile actually occupies.
//
// This is where the extend modes have to happen. A sampler address mode would
// wrap in the atlas rather than in the tile, and the atlas packs a tile against
// its neighbours with `AtlasTile::padding` of zero, so `address::repeat` would
// tile the whole atlas and `address::clamp_to_edge` would clamp to the atlas's
// edge - both of them showing another sprite.
uint extend_texel(int texel, uint extent, BrushExtend extend) {
  int size = (int)extent;
  if (size <= 0) {
    return 0;
  }
  if (extend == BrushExtend_Repeat) {
    int wrapped = texel % size;
    return (uint)(wrapped < 0 ? wrapped + size : wrapped);
  }
  if (extend == BrushExtend_Reflect) {
    int period = size * 2;
    int wrapped = texel % period;
    if (wrapped < 0) {
      wrapped += period;
    }
    return (uint)(wrapped < size ? wrapped : period - 1 - wrapped);
  }
  return (uint)clamp(texel, 0, size - 1);
}

// One bilinear tap of a brush's image, taken by hand.
//
// Four `read`s and two mixes rather than one `sample`, because a single
// hardware tap near a tile edge under `Repeat` or `Reflect` straddles the seam
// and blends in whichever sprite the atlas packed next door. Wrapping the four
// texel indices individually keeps every tap inside the tile, so the seam is
// the seam of the image rather than of the atlas.
float4 sample_path_brush(PathBrush brush, float2 position,
                         texture2d<float> atlas) {
  TransformationMatrix screen_to_brush = brush.screen_to_brush;
  float2 brush_position = float2(
      position.x * screen_to_brush.rotation_scale[0][0] +
          position.y * screen_to_brush.rotation_scale[0][1] +
          screen_to_brush.translation[0],
      position.x * screen_to_brush.rotation_scale[1][0] +
          position.y * screen_to_brush.rotation_scale[1][1] +
          screen_to_brush.translation[1]);

  uint2 tile_size = uint2((uint)brush.tile.bounds.size.width,
                          (uint)brush.tile.bounds.size.height);
  // Brush space measures the image as the unit square; texel centres sit half a
  // texel in from it, which is what the -0.5 is for.
  float2 texel = brush_position * float2(tile_size) - 0.5;
  float2 fraction = fract(texel);
  int2 low = int2(floor(texel));

  uint2 x = uint2(extend_texel(low.x, tile_size.x, brush.x_extend),
                  extend_texel(low.x + 1, tile_size.x, brush.x_extend));
  uint2 y = uint2(extend_texel(low.y, tile_size.y, brush.y_extend),
                  extend_texel(low.y + 1, tile_size.y, brush.y_extend));
  uint2 origin = uint2((uint)brush.tile.bounds.origin.x,
                       (uint)brush.tile.bounds.origin.y);

  float4 top = mix(atlas.read(origin + uint2(x.x, y.x)),
                   atlas.read(origin + uint2(x.y, y.x)), fraction.x);
  float4 bottom = mix(atlas.read(origin + uint2(x.x, y.y)),
                      atlas.read(origin + uint2(x.y, y.y)), fraction.x);
  return mix(top, bottom, fraction.y);
}

// Selects corner radius based on quadrant.
float pick_corner_radius(float2 center_to_point, Corners_ScaledPixels corner_radii) {
  if (center_to_point.x < 0.) {
    if (center_to_point.y < 0.) {
      return corner_radii.top_left;
    } else {
      return corner_radii.bottom_left;
    }
  } else {
    if (center_to_point.y < 0.) {
      return corner_radii.top_right;
    } else {
      return corner_radii.bottom_right;
    }
  }
}

// Signed distance of the point to the quad's border - positive outside the
// border, and negative inside.
float quad_sdf(float2 point, Bounds_ScaledPixels bounds,
               Corners_ScaledPixels corner_radii) {
    float2 half_size = float2(bounds.size.width, bounds.size.height) / 2.0;
    float2 center = float2(bounds.origin.x, bounds.origin.y) + half_size;
    float2 center_to_point = point - center;
    float corner_radius = pick_corner_radius(center_to_point, corner_radii);
    float2 corner_to_point = fabs(center_to_point) - half_size;
    float2 corner_center_to_point = corner_to_point + corner_radius;
    return quad_sdf_impl(corner_center_to_point, corner_radius);
}

float4 pack_bounds(Bounds_ScaledPixels bounds) {
  return float4(bounds.origin.x, bounds.origin.y, bounds.size.width,
                bounds.size.height);
}

float4 pack_corner_radii(Corners_ScaledPixels corner_radii) {
  return float4(corner_radii.top_left, corner_radii.top_right,
                corner_radii.bottom_right, corner_radii.bottom_left);
}

// Coverage of the content mask at this point: 1 inside, 0 outside, antialiased
// across a rounded edge.
//
// A rectangular mask - which is nearly every mask - is already clipped exactly
// by [[clip_distance]] in the vertex stage, so it takes the fast path out.
float content_mask_alpha(float2 point, ContentMask_ScaledPixels mask) {
  return packed_content_mask_alpha(point, pack_bounds(mask.bounds),
                                   pack_corner_radii(mask.corner_radii));
}

// How much of `point` the clip path `clip_id` lets through: one texel of the
// coverage atlas the frame rasterized its masks into.
//
// `tile` is the only part of the window this clip can reach - the content mask
// it was pushed under, intersected with its parent's tile and with the viewport
// - so a point outside it is outside the clip and the atlas is never read
// there.
//
// When `sampled` is zero the clip got no tile and the rectangle *is* the clip.
// That happens when the nesting depth cap or a full atlas refused a mask the
// path did have: the fallback loses the shape but keeps the bound, which is the
// one way to degrade without silently painting the wrong picture. A path with
// no shape to lose - one enclosing no area - is not that case; its tile is
// empty, so the test above has already returned zero and this line is never
// reached for it.
float clip_mask_alpha(float2 point, uint clip_id,
                      constant ClipMask *clip_masks,
                      texture2d<float> clip_atlas) {
  ClipMask mask = clip_masks[clip_id];
  if (point.x < mask.tile.origin.x || point.y < mask.tile.origin.y ||
      point.x >= mask.tile.origin.x + mask.tile.size.width ||
      point.y >= mask.tile.origin.y + mask.tile.size.height) {
    return 0.0;
  }
  if (mask.sampled == 0u) {
    return 1.0;
  }
  float2 atlas_position =
      point + float2(mask.atlas_offset.x, mask.atlas_offset.y);
  return clip_atlas.read(uint2(atlas_position)).r;
}

// `content_mask_alpha` for a mask that arrived as two float4s rather than as a
// struct: `mask_bounds` is (origin.x, origin.y, width, height) and
// `mask_corner_radii` is (top_left, top_right, bottom_right, bottom_left).
// That is the form a fragment stage handed the mask as flat varyings has, and
// it saves those stages a per-fragment load out of the instance buffer.
float packed_content_mask_alpha(float2 point, float4 mask_bounds,
                                float4 mask_corner_radii) {
  if (all(mask_corner_radii == float4(0.0))) {
    return 1.0;
  }

  float2 half_size = mask_bounds.zw / 2.0;
  float2 center = mask_bounds.xy + half_size;
  float2 center_to_point = point - center;

  // A radius wider than half the mask cannot be drawn as written: the four
  // corner arcs would overlap, and picking one of them by quadrant would leave
  // a step where the quadrants meet. `ContentMask::intersect` produces such
  // radii routinely - a radius kept from a tall parent, landing on a short
  // intersection - so clamp here rather than trust the caller. These are the
  // semantics of `Corners::clamp_radii_for_quad_size` on the Rust side.
  float4 radii = min(mask_corner_radii, float4(min(half_size.x, half_size.y)));

  float corner_radius;
  if (center_to_point.x < 0.0) {
    corner_radius = center_to_point.y < 0.0 ? radii.x : radii.w;
  } else {
    corner_radius = center_to_point.y < 0.0 ? radii.y : radii.z;
  }

  float2 corner_center_to_point =
      (fabs(center_to_point) - half_size) + corner_radius;
  return saturate(0.5 - quad_sdf_impl(corner_center_to_point, corner_radius));
}

// Implementation of quad signed distance field
float quad_sdf_impl(float2 corner_center_to_point, float corner_radius) {
    if (corner_radius == 0.0) {
        // Fast path for unrounded corners
        return max(corner_center_to_point.x, corner_center_to_point.y);
    } else {
        // Signed distance of the point from a quad that is inset by corner_radius
        // It is negative inside this quad, and positive outside
        float signed_distance_to_inset_quad =
            // 0 inside the inset quad, and positive outside
            length(max(float2(0.0), corner_center_to_point)) +
            // 0 outside the inset quad, and negative inside
            min(0.0, max(corner_center_to_point.x, corner_center_to_point.y));

        return signed_distance_to_inset_quad - corner_radius;
    }
}

// A standard gaussian function, used for weighting samples
float gaussian(float x, float sigma) {
  return exp(-(x * x) / (2. * sigma * sigma)) / (sqrt(2. * M_PI_F) * sigma);
}

// This approximates the error function, needed for the gaussian integral
float2 erf(float2 x) {
  float2 s = sign(x);
  float2 a = abs(x);
  float2 r1 = 1. + (0.278393 + (0.230389 + (0.000972 + 0.078108 * a) * a) * a) * a;
  float2 r2 = r1 * r1;
  return s - s / (r2 * r2);
}

float blur_along_x(float x, float y, float sigma, float corner,
                   float2 half_size) {
  float delta = min(half_size.y - corner - abs(y), 0.);
  float curved =
      half_size.x - corner + sqrt(max(0., corner * corner - delta * delta));
  float2 integral =
      0.5 + 0.5 * erf((x + float2(-curved, curved)) * (sqrt(0.5) / sigma));
  return integral.y - integral.x;
}

float4 distance_from_clip_rect(float2 unit_vertex, Bounds_ScaledPixels bounds,
                               Bounds_ScaledPixels clip_bounds) {
  float2 position =
      unit_vertex * float2(bounds.size.width, bounds.size.height) +
      float2(bounds.origin.x, bounds.origin.y);
  return float4(position.x - clip_bounds.origin.x,
                clip_bounds.origin.x + clip_bounds.size.width - position.x,
                position.y - clip_bounds.origin.y,
                clip_bounds.origin.y + clip_bounds.size.height - position.y);
}

float4 distance_from_clip_rect_transformed(float2 unit_vertex, Bounds_ScaledPixels bounds,
                               Bounds_ScaledPixels clip_bounds, TransformationMatrix transformation) {
  float2 position =
      unit_vertex * float2(bounds.size.width, bounds.size.height) +
      float2(bounds.origin.x, bounds.origin.y);
  float2 transformed_position = float2(0, 0);
  transformed_position[0] = position[0] * transformation.rotation_scale[0][0] + position[1] * transformation.rotation_scale[0][1];
  transformed_position[1] = position[0] * transformation.rotation_scale[1][0] + position[1] * transformation.rotation_scale[1][1];
  transformed_position[0] += transformation.translation[0];
  transformed_position[1] += transformation.translation[1];

  return float4(transformed_position.x - clip_bounds.origin.x,
                clip_bounds.origin.x + clip_bounds.size.width - transformed_position.x,
                transformed_position.y - clip_bounds.origin.y,
                clip_bounds.origin.y + clip_bounds.size.height - transformed_position.y);
}

float4 over(float4 below, float4 above) {
  float4 result;
  float alpha = above.a + below.a * (1.0 - above.a);
  result.rgb =
      (above.rgb * above.a + below.rgb * below.a * (1.0 - above.a)) / alpha;
  result.a = alpha;
  return result;
}

GradientColor prepare_fill_color(uint tag, uint color_space, Hsla solid,
                                     Hsla color0, Hsla color1) {
  GradientColor out;
  if (tag == 0 || tag == 2 || tag == 3) {
    out.solid = hsla_to_rgba(solid);
  } else if (tag == 1) {
    out.color0 = hsla_to_rgba(color0);
    out.color1 = hsla_to_rgba(color1);

    // Prepare color space in vertex for avoid conversion
    // in fragment shader for performance reasons
    if (color_space == 1) {
      // Oklab
      out.color0 = srgb_to_oklab(out.color0);
      out.color1 = srgb_to_oklab(out.color1);
    }
  }

  return out;
}

float2x2 rotate2d(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float2x2(c, -s, s, c);
}

float4 fill_color(Background background,
                      float2 position,
                      Bounds_ScaledPixels bounds,
                      float4 solid_color, float4 color0, float4 color1) {
  float4 color;

  switch (background.tag) {
    case 0:
      color = solid_color;
      break;
    case 1: {
      // -90 degrees to match the CSS gradient angle.
      float gradient_angle = background.gradient_angle_or_pattern_height;
      float radians = (fmod(gradient_angle, 360.0) - 90.0) * (M_PI_F / 180.0);
      float2 direction = float2(cos(radians), sin(radians));

      // Expand the short side to be the same as the long side
      if (bounds.size.width > bounds.size.height) {
          direction.y *= bounds.size.height / bounds.size.width;
      } else {
          direction.x *=  bounds.size.width / bounds.size.height;
      }

      // Get the t value for the linear gradient with the color stop percentages.
      float2 half_size = float2(bounds.size.width, bounds.size.height) / 2.;
      float2 center = float2(bounds.origin.x, bounds.origin.y) + half_size;
      float2 center_to_point = position - center;
      float t = dot(center_to_point, direction) / length(direction);
      // Check the direction to determine whether to use x or y
      if (abs(direction.x) > abs(direction.y)) {
          t = (t + half_size.x) / bounds.size.width;
      } else {
          t = (t + half_size.y) / bounds.size.height;
      }

      // Adjust t based on the stop percentages
      t = (t - background.colors[0].percentage)
        / (background.colors[1].percentage
        - background.colors[0].percentage);
      t = clamp(t, 0.0, 1.0);

      switch (background.color_space) {
        case 0:
          color = mix(color0, color1, t);
          break;
        case 1: {
          float4 oklab_color = mix(color0, color1, t);
          color = oklab_to_srgb(oklab_color);
          break;
        }
      }

      // Dither to reduce banding in gradients (especially dark/alpha).
      // Triangular-distributed noise breaks up 8-bit quantization steps.
      // ±2/255 for RGB (enough for dark-on-dark compositing),
      // ±3/255 for alpha (needs more because alpha × dark color = tiny steps).
      {
        float2 seed = position * 0.6180339887; // golden ratio spread
        float r1 = fract(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453);
        float r2 = fract(sin(dot(seed, float2(39.3460, 11.135))) * 24634.6345);
        float tri = r1 + r2 - 1.0; // triangular PDF, range [-1, +1]
        color.rgb += tri * 2.0 / 255.0;
        color.a   += tri * 3.0 / 255.0;
      }

      break;
    }
    case 2: {
        float gradient_angle_or_pattern_height = background.gradient_angle_or_pattern_height;
        float pattern_width = (gradient_angle_or_pattern_height / 65535.0f) / 255.0f;
        float pattern_interval = fmod(gradient_angle_or_pattern_height, 65535.0f) / 255.0f;
        float pattern_height = pattern_width + pattern_interval;
        float stripe_angle = M_PI_F / 4.0;
        float pattern_period = pattern_height * sin(stripe_angle);
        float2x2 rotation = rotate2d(stripe_angle);
        float2 relative_position = position - float2(bounds.origin.x, bounds.origin.y);
        float2 rotated_point = rotation * relative_position;
        float pattern = fmod(rotated_point.x, pattern_period);
        float distance = min(pattern, pattern_period - pattern) - pattern_period * (pattern_width / pattern_height) /  2.0f;
        color = solid_color;
        color.a *= saturate(0.5 - distance);
        break;
    }
    case 3: {
        // checkerboard
        float size = background.gradient_angle_or_pattern_height;
        float2 relative_position = position - float2(bounds.origin.x, bounds.origin.y);

        float x_index = floor(relative_position.x / size);
        float y_index = floor(relative_position.y / size);
        float should_be_colored = fmod(x_index + y_index, 2.0);

        color = solid_color;
        color.a *= saturate(should_be_colored);
        break;
    }
  }

  return color;
}
