      module cross_product
      implicit none
      contains
      FUNCTION cross(a, b)
        REAL, DIMENSION(3) :: cross
        REAL, DIMENSION(3), INTENT(IN) :: a, b
        cross(1) = a(2) * b(3) - a(3) * b(2)
        cross(2) = a(3) * b(1) - a(1) * b(3)
        cross(3) = a(1) * b(2) - a(2) * b(1)
      END FUNCTION cross
      end module cross_product
      
      PROGRAM HW3_Prob_2
        use cross_product
        IMPLICIT NONE
        REAL, DIMENSION(3) :: r, f
        REAL, DIMENSION(3) :: m
        ! Part a
        r = [2.0, 5.0, 11.0] ! ft
        f = [0.0, 44.0, -60.0] ! lbs
        m = cross(r, f) ! Cross product
        print*, "Part a ---- r x F = "
        write (*, *) m, "lb-ft"
        
        ! Part b
        r = [1.4, 3.0, 5.7] ! m
        f = [13.0, 25.0, -8.0] ! N
        m = cross(r, f) ! Cross product
        print*, "Part b ---- r x F = "
        write (*, *) m, "N-m"
        
        ! Part c
        r = [-10.0, 15.0, -5.0] ! m
        f = [500.0, -250.0, -300.0] ! N
        m = cross(r, f) ! Cross product
        print*, "Part c ---- r x F = "
        write (*, *) m, "N-m"
      END PROGRAM HW3_Prob_2