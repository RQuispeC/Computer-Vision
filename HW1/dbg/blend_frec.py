if __name__ == "__main__":
    image1 = cv2.imread('input/p1-1-0.jpg', 0)
    image2 = cv2.imread('input/p1-1-1.jpg', 0)
    mask = blending.set_mask(mask_type = 'left_right', img_shape = image1.shape, value = 1)
    image1 = image1 * mask
    image2 = image2 * blending.inverse(mask, value = 1)
    print (image1.shape, image2.shape)
    image1[image1 == 0] = 128
    image2[image2 == 0] = 128
    frec_image1 = fromSpaceToFrequency(image1)
    frec_image2 = fromSpaceToFrequency(image2)
    print (frec_image1.shape, frec_image2.shape)
    level = 2

    lp_a = lp.laplacian_pyramid(change_dim(frec_image1),level)
    lp_b = lp.laplacian_pyramid(change_dim(frec_image2),level)
    lp_a.build()
    lp_b.build()
    
    mid = []
    for i in range(level - 1, -1, -1):
        joint = frec_blending(change_dim_inv(lp_a.get(i)), change_dim_inv(lp_b.get(i)))
        mid.append(change_dim(joint))

    img_blend = mid[0]
    for i in range(1, len(mid)):
       img_blend = lp_a.down(img_blend, mid[i])

    cv2.imwrite('frec_blending5.jpg', fromFrequencyToSpace(change_dim_inv(img_blend)))